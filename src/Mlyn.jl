module Mlyn

include("layer_param_init.jl")
include("layer_activation.jl")
include("layer.jl")

include("model_fwd_pass.jl")
include("model_bck_pass.jl")
include("model_loss.jl")
include("model_optim.jl")
include("batch.jl")
include("model.jl")

export 
  # from layer
  InitMethod,
  method_xavier_uniform,
  method_xavier_normal,
  method_kaiming,

  ActivationMethod,
  method_relu,
  method_sigmoid,
  method_tanh,

  BlueprintLinear,
  BlueprintActivation,

  # from model
  Model,
  create_model,
  train!,
  predict

end
