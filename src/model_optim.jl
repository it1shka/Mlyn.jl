# We keep optimizer separate from the model.
# It is a separate entity that can mutate model's parameters.
# Optimizer may or may not have state: simple SGD is stateless,
# Adam on the other hand stores momentum and velocity

# Stateless optimizer with constant learning rate

@kwdef struct OptimizerSGD
  learning_rate :: LayerFloat
end

function optimize!(linear :: Linear, optimizer :: OptimizerSGD)
  linear.weights .-= optimizer.learning_rate .* linear.grad_weights
  linear.bias .-= optimizer.learning_rate .* linear.grad_bias
end

# TODO: add more optimizers... like Adam

function optimize!(layers :: Vector, optimizer)
  for layer in layers
    optimize!(layer, optimizer)
  end
end

optimize!(@nospecialize(args...); @nospecialize(kwargs...)) = nothing
