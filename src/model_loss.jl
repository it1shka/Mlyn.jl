# We have two distinct functions:
# calculation of loss and calculation of loss gradient
# first - needed purely for tracking the progress
# second - more practical - needed for backward pass

# For regression: Mean-Squared Error
# https://aew61.github.io/blog/artificial_neural_networks/1_background/1.c_loss_functions_and_derivatives.html

function calculate_loss_mse(Y_hat, Y_true)
  N = size(Y_hat, 2)
  diff = Y_hat .- Y_true
  return sum(diff .^ 2) / (2.0 * N)
end

function calculate_loss_gradient_mse(Y_hat, Y_true)
  N = size(Y_hat, 2)
  return (Y_hat .- Y_true) ./ N
end

# For classification: Binary Cross-Entropy
# Formula for the loss function:
# https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/
# Formula for the gradient function (second answer, derivative of the original function):
# https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

function calculate_loss_bce(Y_hat, Y_true)
    N = size(Y_hat, 2)
    ϵ = 1e-7
    Y_hat_clipped = clamp.(Y_hat, ϵ, 1.0 - ϵ)
    term1 = Y_true .* log.(Y_hat_clipped)
    term2 = (1.0 .- Y_true) .* log.(1.0 .- Y_hat_clipped)
    return -sum(term1 .+ term2) / N
end

function calculate_loss_gradient_bce(Y_hat, Y_true)
    N = size(Y_hat, 2)
    ϵ = 1e-7
    Y_hat_clipped = clamp.(Y_hat, ϵ, 1.0 - ϵ)
    numerator = Y_hat_clipped .- Y_true
    denominator = Y_hat_clipped .* (1.0 .- Y_hat_clipped)
    return (numerator ./ denominator) ./ N
end

@enum LossMethod begin
  mse # mean square error
  regression
  bce # binary cross entropy
  classification
end

function get_loss_functions(loss_type :: LossMethod)
  if loss_type == mse || loss_type == regression
    return calculate_loss_mse, calculate_loss_gradient_mse
  end
  return calculate_loss_bce, calculate_loss_gradient_bce
end
