module Mlyn

include("layer_param_init.jl")
include("layer_activation.jl")
include("layer.jl")

include("model_fwd_pass.jl")
include("model_bck_pass.jl")
include("model_loss.jl")
include("model_optim.jl")
include("batch.jl")
include("split.jl")
include("model.jl")
include("utils.jl")

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
  BlueprintBatchNorm1D,

  # from model
  mse,
  regression,
  bce,
  classification,
  LossMethod,
  Model,
  create_model,
  OptimizerSGD,
  OptimizerAdam,
  init_adam!,
  train!,
  predict,
  evaluate_model,
  train_test_split,

  # from utils
  history_plot,
  history_best_result,
  history_plot_test_loss
end
