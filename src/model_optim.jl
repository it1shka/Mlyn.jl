# We keep optimizer separate from the model.
# It is a separate entity that can mutate model's parameters.
# Optimizer may or may not have state: simple SGD is stateless,
# Adam on the other hand stores momentum and velocity

# Stateless optimizer with constant learning rate

@kwdef struct OptimizerSGD
  learning_rate :: LayerFloat
end

function optimize!(layers :: Vector, optimizer :: OptimizerSGD)
  for layer in layers
    if (layer isa Linear) || (layer isa BatchNorm1D)
      optimize!(layer, optimizer)
    end
  end
end

function optimize!(linear :: Linear, optimizer :: OptimizerSGD)
  linear.weights .-= optimizer.learning_rate .* linear.grad_weights
  linear.bias .-= optimizer.learning_rate .* linear.grad_bias
end

function optimize!(batchnorm :: BatchNorm1D, optimizer :: OptimizerSGD)
  batchnorm.γ .-= optimizer.learning_rate .* batchnorm.grad_γ
  batchnorm.β .-= optimizer.learning_rate .* batchnorm.grad_β
end

# Adam optimizer (Adaptive moment estimation)

# https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
# https://eureka.patsnap.com/article/how-does-the-adam-optimizer-work
# Uses ideas of Momentum, AdaGrad and RMSprop.
# Stateful optimizer that takes into account
# previous updates to choose the most optimal learning rate
# for each parameter.
#
# Uses
# * First momentum (mean)
# * Second momentum (variance)

@kwdef mutable struct OptimizerAdam
  α :: LayerFloat = 0.001 # learning rate
  β1 :: LayerFloat = 0.9 # decay rate for mean (momentum 1)
  β2 :: LayerFloat = 0.999 # decay rate for variance (momentum 2)
  ϵ :: LayerFloat = 1e-8 # for numerical stability

  # mean (momentum 1) cache
  m_weights :: Vector{Matrix{LayerFloat}} = []
  m_bias :: Vector{Vector{LayerFloat}} = []
  # variance (momentum 2) cache
  v_weights :: Vector{Matrix{LayerFloat}} = []
  v_bias :: Vector{Vector{LayerFloat}} = []
  # time
  t :: Int = 0
end

function init_adam!(adam :: OptimizerAdam, layers :: Vector)
  adam.m_weights = []
  adam.m_bias = []
  adam.v_weights = []
  adam.v_bias = []

  for layer in layers
    if layer isa Linear
      n_out, n_in = size(layer.weights)
      push!(adam.m_weights, zeros(LayerFloat, n_out, n_in))
      push!(adam.m_bias, zeros(LayerFloat, n_out))
      push!(adam.v_weights, zeros(LayerFloat, n_out, n_in))
      push!(adam.v_bias, zeros(LayerFloat, n_out))
    elseif layer isa BatchNorm1D
      n = size(layer.γ, 1)
      push!(adam.m_weights, zeros(LayerFloat, n, 1))
      push!(adam.m_bias, zeros(LayerFloat, n))
      push!(adam.v_weights, zeros(LayerFloat, n, 1))
      push!(adam.v_bias, zeros(LayerFloat, n))
    end
  end
end

function optimize!(layers :: Vector, optimizer :: OptimizerAdam)
  optimizer.t += 1
  param_index = 1
  for layer in layers
    if (layer isa Linear) || (layer isa BatchNorm1D)
      optimize!(layer, param_index, optimizer)
      param_index += 1
    end
    # TODO: add batchnorm here
  end
end

# formulas are taken from here: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
function optimize!(linear :: Linear, param_index, optimizer :: OptimizerAdam)
  α, β1, β2, ϵ = optimizer.α, optimizer.β1, optimizer.β2, optimizer.ϵ
  t = optimizer.t
  bias_correction_1 = 1 - β1^t
  bias_correction_2 = 1 - β2^t
  optimizer.m_weights[param_index] .= β1 .* optimizer.m_weights[param_index] .+ (1 - β1) .* linear.grad_weights
  optimizer.v_weights[param_index] .= β2 .* optimizer.v_weights[param_index] .+ (1 - β2) .* (linear.grad_weights .^ 2)
  m_hat_w = optimizer.m_weights[param_index] ./ bias_correction_1
  v_hat_w = optimizer.v_weights[param_index] ./ bias_correction_2
  linear.weights .-= α .* m_hat_w ./ (sqrt.(v_hat_w) .+ ϵ)
  optimizer.m_bias[param_index] .= β1 .* optimizer.m_bias[param_index] .+ (1 - β1) .* linear.grad_bias
  optimizer.v_bias[param_index] .= β2 .* optimizer.v_bias[param_index] .+ (1 - β2) .* (linear.grad_bias .^ 2)
  m_hat_b = optimizer.m_bias[param_index] ./ bias_correction_1
  v_hat_b = optimizer.v_bias[param_index] ./ bias_correction_2
  linear.bias .-= α .* m_hat_b ./ (sqrt.(v_hat_b) .+ ϵ)
end

function optimize!(batchnorm :: BatchNorm1D, idx, optimizer :: OptimizerAdam)
  α, β1, β2, ϵ = optimizer.α, optimizer.β1, optimizer.β2, optimizer.ϵ
  t = optimizer.t
  
  bias_correction_1 = 1 - β1^t
  bias_correction_2 = 1 - β2^t
  
  dγ = batchnorm.grad_γ |> x -> reshape(x, :, 1)
  m_γ = optimizer.m_weights[idx]
  v_γ = optimizer.v_weights[idx]
  
  m_γ .= β1 .* m_γ .+ (1 - β1) .* dγ
  v_γ .= β2 .* v_γ .+ (1 - β2) .* (dγ .^ 2)
  
  m_hat_γ = m_γ ./ bias_correction_1
  v_hat_γ = v_γ ./ bias_correction_2
  
  batchnorm.γ .-= α .* vec(m_hat_γ) ./ (sqrt.(vec(v_hat_γ)) .+ ϵ)

  dβ = batchnorm.grad_β
  m_β = optimizer.m_bias[idx]
  v_β = optimizer.v_bias[idx]

  m_β .= β1 .* m_β .+ (1 - β1) .* dβ
  v_β .= β2 .* v_β .+ (1 - β2) .* (dβ .^ 2)
  
  m_hat_β = m_β ./ bias_correction_1
  v_hat_β = v_β ./ bias_correction_2
  
  batchnorm.β .-= α .* m_hat_β ./ (sqrt.(v_hat_β) .+ ϵ)
end
