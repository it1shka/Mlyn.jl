using StatsFuns

function bootstrap(config::LayerConfigLinear)::LayerLinear
  weights = init_params(config.init_method, config.in_features, config.out_features)
  bias = zeros(Float32, config.out_features)
  grad_weights = zeros(Float32, size(weights))
  grad_bias = zeros(Float32, size(bias))
  cached_input = zeros(Float32, 0, config.in_features)
  return LayerLinear(
    config = config,
    weights = weights,
    bias = bias,
    grad_weights = grad_weights,
    grad_bias = grad_bias,
    cached_input = cached_input,
  )
end

function bootstrap(config::LayerConfigActivation)::LayerActivation
  cached_output = zeros(Float32, 0, 0)
  if config.type == act_relu
    return LayerActivation(
      config = config,
      fn = (x) -> x > 0 ? x : 0,
      derivative = (x) -> x > 0 ? 1 : 0,
      cached_output = cached_output,
    )
  end
  if config.type == act_sigmoid
    return LayerActivation(
      config = config,
      fn = logistic,
      derivative = (x) -> logistic(x) * (1 - logistic(x)),
      cached_output = cached_output,
    )
  end
  if config.type == act_tanh
    return LayerActivation(
      config = config,
      fn = tanh,
      derivative = (x) -> sech(x) ^ 2,
      cached_output = cached_output,
    )
  end
  if config.type == act_softmax
    return LayerActivation(
      config = config,
      fn = tanh,
      derivative = (x) -> sech(x) ^ 2,
      cached_output = cached_output,
    )
  end
  throw("Unknown activation function: $(config.type)")
end

function bootstrap(config::LayerConfigBatchnorm)::LayerBatchnorm

end

function bootstrap(config::LayerConfigDropout)::LayerDropout

end
