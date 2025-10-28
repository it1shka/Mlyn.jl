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
