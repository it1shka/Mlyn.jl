function backward_pass!(linear :: Linear, adopted_grad)
  linear.grad_weights .= adopted_grad * linear.cached_input'
  linear.grad_bias .= vec(sum(adopted_grad, dims=2))
  return linear.weights' * adopted_grad
end

function backward_pass!(activation :: Activation, adopted_grad)
  local_grad = activation.derivative(activation.cached_input)
  return adopted_grad .* local_grad
end

function backward_pass!(layers :: Vector, initial_grad)
  running_grad = initial_grad
  for layer in reverse(layers)
    running_grad = backward_pass!(layer, running_grad)
  end
end
