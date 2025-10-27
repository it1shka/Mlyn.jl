# Later we can change that datatype
const LayerFloat = Float32

# Linear layer

@kwdef mutable struct Linear
  weights :: Matrix{LayerFloat}
  bias :: Vector{LayerFloat}
  cached_input :: Matrix{LayerFloat}
  grad_weights :: Matrix{LayerFloat}
  grad_bias :: Vector{LayerFloat}
end

@kwdef struct BlueprintLinear
  n_in :: Int
  n_out :: Int
  init_method :: InitMethod
end

function create(blueprint :: BlueprintLinear)
  return Linear(
    weights = init_params(blueprint.init_method, blueprint.n_in, blueprint.n_out),
    bias = zeros(LayerFloat, blueprint.n_out),
    cached_input = Matrix{LayerFloat}(undef, 0, 0),
    grad_weights = zeros(LayerFloat, blueprint.n_out, blueprint.n_in),
    grad_bias = zeros(LayerFloat, blueprint.n_out)
  )
end

# Activation layer

@kwdef mutable struct Activation
  fn :: Function
  derivative :: Function
  cached_input :: Matrix{LayerFloat}
end

@kwdef struct BlueprintActivation
  act_method :: ActivationMethod
end

function create(blueprint :: BlueprintActivation)
  return Activation(
    fn = get_activation(blueprint.act_method),
    derivative = get_activation_derivative(blueprint.act_method),
    cached_input = Matrix{LayerFloat}(undef, 0, 0)
  )
end
