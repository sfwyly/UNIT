
from utils import *
import tensorflow as tf
import os
from networks import *

# Trainer
class UNIT_Trainer():

    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        #self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        #dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        #gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

        # self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        #
        # # Network weight initialization
        # self.apply(weights_init(hyperparameters['init']))
        # self.dis_a.apply(weights_init('gaussian'))
        # self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        style_layers = ['block1_pool', 'block2_pool', 'block3_pool']
        if ('vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0):
            self.vgg = self.vgg_layers(style_layers)


    def vgg_layers(self, layer_names):

        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)

        return model
    def recon_criterion(self, input, target):
        input = tf.cast(input, tf.float32)
        target = tf.cast(target, tf.float32)
        return tf.reduce_mean(tf.abs(input - target))

    def __compute_kl(self, mu):
        mu = tf.cast(mu, tf.float32)
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = tf.pow(mu, 2)
        encoding_loss = tf.reduce_mean(mu_2)

        return encoding_loss

    def call(self, x_a, x_b):
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):

        # self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # encode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon)
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon)

        # mask image -> true image
        # true image -> mask image is don't process

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)

        self.loss_encode_a_b = self.recon_criterion(h_a, h_b)

        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) #
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b) #
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba) # true -> mask
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invarient perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_a) # true -> mask
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_b)
        # total loss
        self.loss_gen_total = 0 * hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              0 * hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              0 * hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              0 * hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              0 * hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              0 * hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        return self.loss_gen_total

    def compute_vgg_loss(self, vgg, img, target):

        img_vgg = img #vgg_preprocess(img)
        target_vgg = target #vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        loss = 0.
        for img_f, target_f in zip(img_fea, target_fea):
            loss += tf.reduce_mean(tf.abs(img_f - target_f)**2)
        return loss
    # calc style loss
    def compute_style_loss(self, vgg, img, target):
        img_vgg = img #vgg_preprocess(img)
        target_vgg = target #vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        loss = 0.
        for img_f, target_f in zip(img_fea, target_fea):
            b, h, w, c = img_f.shape
            img_n = tf.linalg.einsum("bijc,bijd->bcd", img_f, img_f)/(h*w*c)
            target_n = tf.linalg.einsum("bijc,bijd->bcd", target_f, target_f)/(h*w*c)

            loss += tf.reduce_mean((img_f - target_f)**2)
        return loss


    def dis_update(self, x_a, x_b, hyperparameters):

        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba, x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab, x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        # train
        return self.loss_dis_total
    def save_weights(self):

        self.gen_a.save_weights("/root/gen_a.h5")
        self.gen_b.save_weights("/root/gen_b.h5")
        self.dis_a.save_weights("/root/dis_a.h5")
        self.dis_b.save_weights("/root/dis_b.h5")
    def load_weights(self):
        self.gen_a.load_weights("/root/gen_a.h5")
        self.gen_b.load_weights("/root/gen_b.h5")
        self.dis_a.load_weights("/root/dis_a.h5")
        self.dis_b.load_weights("/root/dis_b.h5")

configs = get_config("configs/unit_edges2handbags_folder.yaml")

gen_optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001, beta_1= 0.5, beta_2= 0.999)
dis_optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001, beta_1= 0.5, beta_2= 0.999)

@tf.function()
def trainer_step(mask_image_list, image_list, mask_list, unit_trainer):

    with tf.GradientTape() as gen_tape , tf.GradientTape() as dis_tape:

        loss_gen_total = unit_trainer.gen_update(mask_image_list, image_list, configs)
        loss_dis_total = unit_trainer.dis_update(mask_image_list, image_list, configs)

    gen_grads = gen_tape.gradient(loss_gen_total, (unit_trainer.gen_a.trainable_variables+ unit_trainer.gen_b.trainable_variables))
    gen_optimizer.apply_gradients(zip(gen_grads, (unit_trainer.gen_a.trainable_variables+ unit_trainer.gen_b.trainable_variables)))

    dis_grads = dis_tape.gradient(loss_dis_total,
                                  (unit_trainer.dis_a.trainable_variables+ unit_trainer.dis_b.trainable_variables))
    dis_optimizer.apply_gradients(
        zip(dis_grads, (unit_trainer.dis_a.trainable_variables+ unit_trainer.dis_b.trainable_variables)))

    return loss_gen_total, loss_dis_total





