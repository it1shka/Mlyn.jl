function forward_pass!(linear :: Linear, X, train)
  if train
    linear.cached_input = X
  end
  return linear.weights * X .+ linear.bias
end

function forward_pass!(activation :: Activation, X, train)
  if train
    activation.cached_input = X
  end
  return activation.fn(X)
end

function forward_pass!(batchnorm :: BatchNorm1D, X, train)
  ϵ = LayerFloat(1e-5)

  if train
    μ_batch = mean(X, dims=2) |> vec
    σ2_batch = var(X, dims=2, corrected=false) |> vec
    
    batchnorm.μ_running .= batchnorm.momentum .* batchnorm.μ_running .+ (1 - batchnorm.momentum) .* μ_batch
    batchnorm.σ2_running .= batchnorm.momentum .* batchnorm.σ2_running .+ (1 - batchnorm.momentum) .* σ2_batch
    
    X_centered = X .- μ_batch
    X_std = sqrt.(σ2_batch .+ ϵ)
    X_hat = X_centered ./ X_std
    
    batchnorm.X_hat = X_hat
    batchnorm.μ_batch = μ_batch
    batchnorm.σ2_batch = σ2_batch
    
  else
    X_centered = X .- batchnorm.μ_running
    X_std = sqrt.(batchnorm.σ2_running .+ ϵ)
    X_hat = X_centered ./ X_std
  end

  Y = batchnorm.γ .* X_hat .+ batchnorm.β
  return Y
end

function forward_pass!(layers :: Vector, X; train = true)
  output = X
  for layer in layers
    output = forward_pass!(layer, output, train)
  end
  return output
end
