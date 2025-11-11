using Plots

function history_plot(history)
  test_loss = getfield.(history, 1)
  train_loss = getfield.(history, 2)
  plot([test_loss train_loss], title="Learning curves", label=["test loss" "train loss"], linewidth=2)
end

function history_plot_test_loss(history)
  test_loss = getfield.(history, 1)
  plot(test_loss, title="Learning curve: test loss", label="test loss")
end

function history_plot_gradients(history)
  gradients = getfield.(history, 3)
  plot(gradients, title="Gradients", label="gradient")
end

function history_best_result(history)
  test_loss = getfield.(history, 1)
  train_loss = getfield.(history, 2)
  return (minimum(test_loss), minimum(train_loss))
end
