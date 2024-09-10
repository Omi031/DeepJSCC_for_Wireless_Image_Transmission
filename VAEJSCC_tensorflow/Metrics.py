import tensorflow as tf
from tensorflow.keras import metrics


class VAETrainMetrics:
    def __init__(self):
        self.rc_loss = metrics.Mean(name="train_rc_loss")
        self.kl_loss = metrics.Mean(name="train_kl_loss")
        self.loss = metrics.Mean(name="train_loss")

    def update_on_train(self, rc_loss, kl_loss, loss):
        self.rc_loss.update_state(rc_loss)
        self.kl_loss.update_state(kl_loss)
        self.loss.update_state(loss)

    def update_on_test(self, result):
        pass

    def reset(self):
        self.rc_loss.reset_states()
        self.kl_loss.reset_states()
        self.loss.reset_states()

    def display(self, epoch, psnr, lpips=None):
        template = {
            "Epoch {}, PSNR: {:.2f}, LPIPS: {:.4g}, RC Loss: {:.2g}, KL Loss: {:.2g}, Loss: {:.2g}"
        }
        print(
            template.format(
                epoch,
                psnr,
                lpips,
                self.rc_loss(),
                self.kl_loss(),
                self.loss(),
            )
        )
