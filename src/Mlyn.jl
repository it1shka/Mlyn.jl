module Mlyn

include("layer_param_init.jl")
include("layer_activation.jl")
include("layer.jl")

export 
  InitMethod,
  method_xavier_uniform,
  method_xavier_normal,
  method_kaiming,
  ActivationMethod,
  method_relu,
  method_sigmoid,
  method_tanh,
  method_softmax,
  BlueprintLinear,
  BlueprintActivation

include("model_fwd_pass.jl")
include("model.jl")

export
  create_model,
  predict

end
