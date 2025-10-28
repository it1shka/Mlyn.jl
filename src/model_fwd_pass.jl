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

function forward_pass!(layers :: Vector, X; train = true)
  output = X
  for layer in layers
    output = forward_pass!(layer, output, train)
  end
  return output
end
