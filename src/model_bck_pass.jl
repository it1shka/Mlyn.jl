# Formula for weights (at the bottom of the paper):
# https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
# Formula for bias:
# https://robotchinwag.com/posts/linear-layer-deriving-the-gradient-for-the-backward-pass/

# On different sites you can find versions with
# different order of matrix multiplication.
# Order of matrix multiplication depends on the
# convention. In my convention, columns are samples

function backward_pass!(linear :: Linear, adopted_grad)
  linear.grad_weights .= adopted_grad * linear.cached_input'
  linear.grad_bias .= vec(sum(adopted_grad, dims=2))
  return linear.weights' * adopted_grad
end

function backward_pass!(activation :: Activation, adopted_grad)
  local_grad = activation.derivative(activation.cached_input)
  return adopted_grad .* local_grad
end

function backward_pass!(batchnorm :: BatchNorm1D, adopted_grad)
    N = size(adopted_grad, 2)
    ϵ = LayerFloat(1e-5)
    
    X_hat = batchnorm.X_hat
    σ2_batch = batchnorm.σ2_batch
    γ = batchnorm.γ
    
    batchnorm.grad_β .= vec(sum(adopted_grad, dims=2)) 
    batchnorm.grad_γ .= vec(sum(adopted_grad .* X_hat, dims=2))
    
    dY_scaled = γ .* adopted_grad
    var_term = sum(dY_scaled .* X_hat, dims=2)
    mean_term = sum(dY_scaled, dims=2)
    inv_std = 1.0 ./ sqrt.(σ2_batch .+ ϵ)
    dX = (inv_std) .* (dY_scaled .- (1/N) .* mean_term .- (1/N) .* (X_hat .* var_term))
    
    return dX
end

function backward_pass!(layers :: Vector, initial_grad)
  running_grad = initial_grad
  for layer in reverse(layers)
    running_grad = backward_pass!(layer, running_grad)
  end
end
