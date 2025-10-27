@enum ActivationMethod begin
  method_relu
  method_sigmoid
  method_tanh
end

act_relu(X) = @. max(X, 0)
act_sigmoid(X) = @. 1.0 / (1 + exp(-X)) 
act_tanh(X) = @. tanh(X)

function get_activation(method :: ActivationMethod)
    if method == method_relu
        return act_relu
    end
    if method == method_sigmoid
        return act_sigmoid
    end
    return act_tanh
end

# Derivative functions (all accept the cached input Z and return the local gradient f'(Z))
relu_derivative(Z) = @. ifelse(Z > 0, 1.0, 0.0)

function sigmoid_derivative(Z)
    A = act_sigmoid(Z)
    return A .* (1.0 .- A)
end

function tanh_derivative(Z)
    A = act_tanh(Z)
    return 1.0 .- (A .* A)
end

function get_activation_derivative(method :: ActivationMethod)
    if method == method_relu
        return relu_derivative
    end
    if method == method_sigmoid
        return sigmoid_derivative
    end
    return tanh_derivative
end
