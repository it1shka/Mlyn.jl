@kwdef mutable struct LayerLinear
  config::LayerConfigLinear
  weights::Matrix{Float32}
  bias::Vector{Float32}
  grad_weights::Matrix{Float32}
  grad_bias::Vector{Float32}
  cached_input::Matrix{Float32}
end

@kwdef mutable struct LayerActivation
  config::LayerConfigActivation
  fn::Function
  derivative::Function
  cached_output::Matrix{Float32}
end

@kwdef mutable struct LayerBatchnorm
  config::LayerConfigBatchnorm
  # TODO: implement
end

@kwdef mutable struct LayerDropout
  config::LayerConfigDropout
  # TODO:
end
