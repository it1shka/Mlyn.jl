# Later we can change that datatype
const LayerFloat = Float32

# Linear layer

@kwdef struct Linear
  weights :: Matrix{LayerFloat}
  bias :: Vector{LayerFloat}
end

@kwdef struct BlueprintLinear
  n_in :: Int
  n_out :: Int
  init_method :: InitMethod
end

function create(blueprint :: BlueprintLinear)
  return Linear(
    weights = init_params(blueprint.init_method, blueprint.n_in, blueprint.n_out),
    bias = zeros(LayerFloat, blueprint.n_out)
  )
end

# Activation layer

@kwdef struct Activation
  fn :: Function
end

@kwdef struct BlueprintActivation
  act_method :: ActivationMethod
end

function create(blueprint :: BlueprintActivation)
  return Activation(
    fn = get_activation(blueprint.act_method)
  )
end
