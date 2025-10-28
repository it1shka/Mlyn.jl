@kwdef struct Model
  layers :: Vector
  loss_fn :: Function
  loss_grad_fn :: Function
end

function create_model(loss_method :: LossMethod, layer_blueprints :: Vector)
  loss_fn, loss_grad_fn = get_loss_functions(loss_method)
  layers = create.(layer_blueprints)
  return Model(layers, loss_fn, loss_grad_fn)
end

function zero_grad!(layer :: Linear)
  layer.grad_weights .= 0.0
  layer.grad_bias .= 0.0
end

function zero_grad!(layers :: Vector)
  for layer in layers
    zero_grad!(layer)
  end
end

zero_grad!(@nospecialize(args...); @nospecialize(kwargs...)) = nothing

function train_batch!(model :: Model, optimizer, X_batch, Y_batch)
  zero_grad!(model.layers)
  Y_hat = forward_pass!(model.layers, X_batch)
  loss = model.loss_fn(Y_hat, Y_batch)
  initial_grad = model.loss_grad_fn(Y_hat, Y_batch)
  backward_pass!(model.layers, initial_grad)
  optimize!(model.layers, optimizer)
  return loss
end

function evaluate_model(model :: Model, X, Y_true)
  Y_hat = forward_pass!(model.layers, X; train=false)
  loss = model.loss_fn(Y_hat, Y_true)
  return loss
end

function train!(model :: Model, optimizer, epochs, X, Y, X_test, Y_test; batch_size = 32, logging = true)
  loss_history = []
  for epoch in 1:epochs
    batches = create_batches(X, Y, batch_size)
    total_loss = 0.0
    for (X_batch, Y_batch) in batches
      batch_loss = train_batch!(model, optimizer, X_batch, Y_batch)
      total_loss += batch_loss
    end
    avg_batch_loss = total_loss / length(batches)
    test_loss = evaluate_model(model, X_test, Y_test)
    push!(loss_history, (test_loss, avg_batch_loss))
    if logging
      println("[EPOCH $(epoch)] Test loss = $(test_loss); Learning loss = $(avg_batch_loss)")
    end
  end
  return loss_history
end

function predict(model :: Model, X)
  return forward_pass!(model.layers, X; train = false)
end
