@enum ActivationMethod begin
  method_relu
  method_sigmoid
  method_tanh
  method_softmax
end

# All functions should accept a tensor
# since some of them (like softmax) may want
# to aggregate information about the whole tensor
# (like a mean value, for example)

relu(X) = @. max(X, 0)
sigmoid(X) = @. 1.0 / (1 + exp(-X)) # TODO: numerical stability
tanh(X) = @. tanh(X)
function softmax(X) # TODO: allow matrices
  X_shift = X .- maximum(X)
  expX = exp.(X_shift)
  return expX ./ sum(expX)
end

function get_activation(method :: ActivationMethod)
  if method == method_relu
    return relu
  end
  if method == method_sigmoid
    return sigmoid
  end
  if method == method_tanh
    return tanh
  end
  return softmax
end
