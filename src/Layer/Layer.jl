module Layer
include("configs.jl")
include("param_init.jl")
include("layers.jl")
include("bootstrap.jl")

export ActivationFunctionType,
  ParamInitializationMethod,
  LayerConfigLinear,
  LayerConfigActivation,
  LayerConfigBatchnorm,
  LayerConfigDropout

export LayerLinear, LayerActivation, LayerBatchnorm, LayerDropout

export bootstrap
end
