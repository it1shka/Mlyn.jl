function backward_pass(linear :: Linear, adopted_grad)

end

function backward_pass(activation :: Activation, adopted_grad)

end

function backward_pass(layers :: Vector, initial_grad)
  running_grad = initial_grad
  for layer in reverse(layers)
    running_grad = backward_pass(layer, running_grad)
  end
end
