using Random

function train_test_split(X, Y, percentile)
  n_samples = size(X, 2)
  shuffled_indices = shuffle(1:n_samples)
  delimiter = (n_samples * percentile) |> trunc |> Int
  X_shuffled, Y_shuffled = X[:, shuffled_indices], Y[:, shuffled_indices]
  X_train, X_test = X_shuffled[:, 1:delimiter], X_shuffled[:, delimiter+1:n_samples]
  Y_train, Y_test = Y_shuffled[:, 1:delimiter], Y_shuffled[:, delimiter+1:n_samples]
  return X_train, Y_train, X_test, Y_test
end
