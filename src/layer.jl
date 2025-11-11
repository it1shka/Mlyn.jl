# Later we can change that datatype
const LayerFloat = Float32

# Linear layer

@kwdef mutable struct Linear
  weights :: Matrix{LayerFloat}
  bias :: Vector{LayerFloat}
  cached_input :: Matrix{LayerFloat}
  grad_weights :: Matrix{LayerFloat}
  grad_bias :: Vector{LayerFloat}
end

@kwdef struct BlueprintLinear
  n_in :: Int
  n_out :: Int
  init_method :: InitMethod
end

function create(blueprint :: BlueprintLinear)
  return Linear(
    weights = init_params(blueprint.init_method, blueprint.n_in, blueprint.n_out),
    bias = zeros(LayerFloat, blueprint.n_out),
    cached_input = Matrix{LayerFloat}(undef, 0, 0),
    grad_weights = zeros(LayerFloat, blueprint.n_out, blueprint.n_in),
    grad_bias = zeros(LayerFloat, blueprint.n_out)
  )
end

# Activation layer

@kwdef mutable struct Activation
  fn :: Function
  derivative :: Function
  cached_input :: Matrix{LayerFloat}
end

@kwdef struct BlueprintActivation
  act_method :: ActivationMethod
end

function create(blueprint :: BlueprintActivation)
  return Activation(
    fn = get_activation(blueprint.act_method),
    derivative = get_activation_derivative(blueprint.act_method),
    cached_input = Matrix{LayerFloat}(undef, 0, 0)
  )
end

# Batch normalization layer

# https://www.geeksforgeeks.org/deep-learning/what-is-batch-normalization-in-deep-learning/

@kwdef mutable struct BatchNorm1D
  # Trainable Parameters: y = γx̂+β
  # updated by optimizer, depend on grad_γ and grad_β
  γ :: Vector{LayerFloat} # scale
  β :: Vector{LayerFloat} # shift
  # Gradients for optimizer
  grad_γ :: Vector{LayerFloat}
  grad_β :: Vector{LayerFloat}

  # parameters used only in non-training mode
  μ_running :: Vector{LayerFloat} # mean of all batches
  σ2_running :: Vector{LayerFloat} # variance of all batches
  momentum :: LayerFloat # Decay rate for running averages (often 0.9 or 0.99)

  # Cached values for backward pass
  X_hat :: Matrix{LayerFloat} # Normalized input
  μ_batch :: Vector{LayerFloat} # Batch mean
  σ2_batch :: Vector{LayerFloat} # Batch variance

  # TODO: move epsilon here ϵ for consistency
end

@kwdef struct BlueprintBatchNorm1D
  n_features :: Int
  momentum :: LayerFloat = 0.9
end

function create(blueprint :: BlueprintBatchNorm1D)
  n = blueprint.n_features
  return BatchNorm1D(
    γ = ones(LayerFloat, n),
    β = zeros(LayerFloat, n),
    grad_γ = zeros(LayerFloat, n),
    grad_β = zeros(LayerFloat, n),
    μ_running = zeros(LayerFloat, n),
    σ2_running = ones(LayerFloat, n), # Initialize variance to 1.0
    momentum = blueprint.momentum,
    X_hat = Matrix{LayerFloat}(undef, 0, 0),
    μ_batch = Vector{LayerFloat}(undef, 0),
    σ2_batch = Vector{LayerFloat}(undef, 0)
  )
end
