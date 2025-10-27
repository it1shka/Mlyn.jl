using Distributions: Uniform, Normal

function init_params_xavier_uniform(n_in, n_out)
  x = √(6 / (n_in + n_out))
  return rand(Uniform(-x, x), n_out, n_in)
end

function init_params_xavier_normal(n_in, n_out)
  σ = √(2 / (n_in + n_out))
  return rand(Normal(0.0, σ), n_out, n_in)
end

function init_params_kaiming(n_in, n_out)
  σ = √(2 / n_in)
  return rand(Normal(0.0, σ), n_out, n_in)
end

@enum InitMethod begin
  method_xavier_uniform
  method_xavier_normal
  method_kaiming
end

function init_params(init_method :: InitMethod, n_in, n_out)
  if init_method == method_xavier_uniform
    return init_params_xavier_uniform(n_in, n_out)
  end
  if init_method == method_xavier_normal
    return init_params_xavier_normal(n_in, n_out)
  end
  return init_params_kaiming(n_in, n_out)
end
