import tensorflow as tf

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
  def __init__(self, iteration_change, lr):
    super(CustomLearningRateScheduler, self).__init__()
    self.iteration = 0
    self.iteration_change = iteration_change
    self.lr = lr
  def on_batch_begin(self, batch, logs=None):
    self.iteration += 1
    if self.iteration == self.iteration_change:
      tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
      print(f"\nIteration {self.iteration}: Learning rate is {self.lr}.")