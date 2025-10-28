using Random: shuffle

function create_batches(X, Y, batch_size)
  n_samples = size(X, 2)
  batches = []
  shuffled_indices = shuffle(1:n_samples)
  for i in 1:batch_size:n_samples
    end_index = min(i + batch_size - 1, n_samples)
    batch_indices = shuffled_indices[i : end_index]
    X_batch = X[:, batch_indices]
    Y_batch = Y[:, batch_indices]
    push!(batches, (X_batch, Y_batch))
  end
  return batches
end
