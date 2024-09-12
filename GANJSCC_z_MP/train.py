import tensorflow as tf
from metrics import PSNR, LPIPS, tf_tensor2pt_tensor

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def dis_loss_fn(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def gen_loss_fn(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


class Trainer:
    def __init__(self, gen, dis, gen_optim, dis_optim, val=[-1, 1]):
        super(Trainer, self).__init__()
        self.gen = gen
        self.dis = dis
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        max_val = val[1] - val[0]
        self.psnr_fn = PSNR(val_pre=[-1, 1], val_new=[0, 1])
        self.lpips_fn = LPIPS(val_pre=[-1, 1], val_new=[-1, 1])

    @tf.function
    def train_step(self, true):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            pred = self.gen(true)
            true_dis = self.dis(true)
            pred_dis = self.dis(pred)

            gen_loss = gen_loss_fn(pred_dis)
            dis_loss = dis_loss_fn(true_dis, pred_dis)

        gen_grads = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss, self.dis.trainable_variables)
        self.gen_optim.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        self.dis_optim.apply_gradients(zip(dis_grads, self.dis.trainable_variables))

        return gen_loss, dis_loss

    def test_step(self, true):
        pred = self.gen(true)
        true_dis = self.dis(true)
        pred_dis = self.dis(pred)

        gen_loss = gen_loss_fn(pred_dis)
        dis_loss = dis_loss_fn(true_dis, pred_dis)
        psnr = tf.reduce_mean(self.psnr_fn(true, pred))
        lpips = tf.reduce_mean(
            self.lpips_fn(tf_tensor2pt_tensor(true), tf_tensor2pt_tensor(pred))
            .detach()
            .numpy()
        )

        return gen_loss, dis_loss, psnr, lpips

    def predict(self, true):
        pred = self.gen(true)
        return pred
