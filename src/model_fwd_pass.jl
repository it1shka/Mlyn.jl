function forward_pass!(linear :: Linear, X)
  linear.cached_input = X
  return linear.weights * X .+ linear.bias
end

function forward_pass!(activation :: Activation, X)
  activation.cached_input = X
  return activation.fn(X)
end

function forward_pass!(layers :: Vector, X)
  output = X
  for layer in layers
    output = forward_pass!(layer, output)
  end
  return output
end
