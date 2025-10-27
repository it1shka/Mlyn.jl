@enum ActivationFunctionType begin
  act_relu
  act_sigmoid
  act_tanh
  act_softmax
end

@enum ParamInitializationMethod begin
  init_zero
  init_xavier_uniform
  init_xavier_normal
  init_he
end

@kwdef struct LayerConfigLinear
  in_features::Int
  out_features::Int
  bias::Bool
  init_method::ParamInitializationMethod
end

@kwdef struct LayerConfigActivation
  type::ActivationFunctionType
end

@kwdef struct LayerConfigBatchnorm
  # TODO: implement
end

@kwdef struct LayerConfigDropout
  p::Float32
end
