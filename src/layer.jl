@enum LayerType begin
  linear
  relu
  sigmoid
  tanh
  softmax
  batchnorm
  dropout
end

struct Layer
  type :: LayerType
  input_dim :: Int
  output_dim :: Int
end

macro linear(in, out)
  return :(Layer(linear, in, out))
end

macro relu(in, out)
  return :(Layer(relu, in, out))
end

macro sigmoid(in, out)
  return :(Layer(sigmoid, in, out))
end

macro tanh(in, out)
  return :(Layer(tanh, in, out))
end

macro softmax(in, out)
  return :(Layer(softmax, in, out))
end

macro batchnorm(in, out)
  return :(Layer(batchnorm, in, out))
end

macro dropout(in, out)
  return :(Layer(dropout, in, out))
end

struct ImplLinearLayer
  input :: Matrix{Float32}
  weights :: Matrix{Float32}
  bias :: Vector{Float32}
  output :: Matrix{Float32}
  weights_gradients :: Matrix{Float32}
  bias_gradients :: Vector{Float32}
end

struct ImplActivationLayer
  
end

function create_linear_layer(layer :: Layer)
  return ImplLinearLayer(;
    input=
  )
end

function create_relu_layer(layer :: Layer)

end

function create_sigmoid_layer(layer :: Layer)

end

function create_tanh_layer(layer :: Layer)

end

function create_softmax_layer(layer :: Layer)

end

function create_batchnorm_layer(layer :: Layer)

end

function create_dropout_layer(layer :: Layer)

end

function create_layer(layer :: Layer)
  @assert layer.input_dim > 0
  @assert layer.output_dim > 0
  if layer.type == linear
    return create_linear_layer(layer)
  elseif layer.type == relu
    return create_relu_layer(layer)
  elseif layer.type == sigmoid
    return create_sigmoid_layer(layer)
  elseif layer.type == tanh
    return create_tanh_layer(layer)
  elseif layer.type == softmax
    return create_softmax_layer(layer)
  elseif layer.type == batchnorm
    return create_batchnorm_layer(layer)
  elseif layer.type == dropout
    return create_dropout_layer(layer)
  else
    throw(ArgumentError("unexpected layer type: $(string(layer.type))"))
  end
end

