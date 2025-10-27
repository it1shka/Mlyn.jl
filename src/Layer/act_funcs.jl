function relu(X::Vector{Float32})::Vector{Float32}
  return (x -> x > 0 : x : 0).(X)
end

