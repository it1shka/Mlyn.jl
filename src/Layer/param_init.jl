using Distributions

function init_params(
  method::ParamInitializationMethod,
  in_feat::Int,
  out_feat::Int,
)::Matrix{Float32}
  if method == init_zero
    return init_params_zero(in_feat, out_feat)
  end
  if method == init_xavier_uniform
    return init_params_xavier_uniform(in_feat, out_feat)
  end
  if method == init_xavier_normal
    return init_params_xavier_normal(in_feat, out_feat)
  end
  if method == init_he
    return init_params_he(in_feat, out_feat)
  end
  throw("Unexpected param initialization method: $(method)")
end

function init_params_zero(in_feat::Int, out_feat::Int)::Matrix{Float32}
  return zeros(Float32, out_feat, in_feat)
end

# Source for Xavier initialization:
# https://www.geeksforgeeks.org/deep-learning/xavier-initialization/

function init_params_xavier_uniform(in_feat::Int, out_feat::Int)::Matrix{Float32}
  σ = √(6.0 / (in_feat + out_feat))
  return rand(Uniform(-σ, σ), out_feat, in_feat)
end

function init_params_xavier_normal(in_feat::Int, out_feat::Int)::Matrix{Float32}
  σ = √(2 / (in_feat + out_feat))
  return rand(Normal(0.0, σ), out_feat, in_feat)
end

# Source for He Kaiming initialization:
# www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/

function init_params_he(in_feat::Int, out_feat::Int)::Matrix{Float32}
  σ = √(2 / in_feat)
  return rand(Normal(0.0, σ), out_feat, in_feat)
end
