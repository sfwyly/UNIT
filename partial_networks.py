

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import numpy as np

# 进一步 进行部分卷积操作
class PConv(layers.Layer):

    def __init__(self, kernel=3, dilation_rate=1, strides=2, in_channels=64, out_channels=64, activation="relu",
                 has_bn=True, is_partial=True, use_bias = False, training=True, **kwargs):
        super(PConv, self).__init__(**kwargs)
        self.slide_window = kernel ** 2
        self.kernel = kernel
        self.strides = strides
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dilation_rate = dilation_rate

        if (activation == "relu"):
            self.activation = tf.nn.relu
        elif (activation == "leaky_relu"):
            self.activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            self.activation = tf.nn.elu
        else:
            self.activation = tf.nn.relu

        self.eps = 1e-8
        self.use_bias = use_bias
        self.conv = tf.keras.layers.Conv2D(self.out_channels, kernel_size=self.kernel,
                                             dilation_rate=self.dilation_rate, strides=self.strides, activation = None, padding="same",
                                             use_bias=False, trainable = True)  #
        self.has_bn = has_bn
        self.is_partial = is_partial

        # use bias
        if(self.use_bias):
            self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
        # has or not bn
        if (self.has_bn):
            self.bn = layers.BatchNormalization()
        # is partial conv
        if(self.is_partial):
            self.weights_updater = tf.ones((self.kernel, self.kernel, self.in_channels, self.out_channels))
        self.training = training

    def call(self, input, mask):

        if(self.is_partial):

            update_mask = tf.nn.conv2d(mask, self.weights_updater, strides=self.strides, padding="SAME")
            mask_ratio = (self.slide_window * self.in_channels) / (update_mask + self.eps)
            update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
            mask_ratio = mask_ratio * update_mask

        output = self.conv(input)  # -self.bias

        if (self.is_partial):
            if(self.use_bias):
                output = (output - self.bias) * mask_ratio + self.bias
            else:
                output = output * mask_ratio
            output = output * update_mask
        else:
            if (self.use_bias):
                output = self.activation(output) + self.bias
            else:
                output = self.activation(output)
        # return
        if(self.is_partial):

            if (self.has_bn):
                return self.activation(self.bn(output, training=self.training)), update_mask
            else:
                return self.activation(output), update_mask
        else:
            if(self.has_bn):
                return self.activation(self.bn(output, training=self.training))
            else:
                return output # 前面已经处理好了

class VaeGan(layers.Layer):

    def __init__(self, in_channels, out_channels,n_layers):
        super(VaeGan,self).__init__()
        self.pConv1 = PConv(kernel=3, dilation_rate=1, strides=2, in_channels = in_channels, out_channels = out_channels, activation="relu",
                 has_bn=True, is_partial=True, use_bias = False, training=True)

        self.model = []

        self.pConv2 = PConv(kernel=3, dilation_rate=1, strides=2, in_channels = out_channels, out_channels = out_channels, activation="relu",
                 has_bn=True, is_partial=True, use_bias = False, training=True)


    def call(self, input, mask):

        self.pConv1(input, mask)

        pass

class LinearBlock(layers.Layer):

    def __init__(self, input_dim, output_dim, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if (norm != 'none'):
            self.norm = layers.BatchNormalization()
        if (activation != 'none'):
            self.activation = layers.Activation(activation)
        self.fc = layers.Dense(output_dim, use_bias=self.use_bias)

    def call(self, x):
        out = self.fc(x)
        if (self.norm):
            out = self.norm(out)
        if (self.activation):
            out = self.activation(out)
        return out

class MLP(layers.Layer):

    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]
        self.model = models.Sequential(self.model)

    def call(self, x):
        return self.model(x)


class Conv2dBlock(layers.Layer):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='none', activation='relu',
                 pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if (norm != 'none'):
            self.norm = layers.BatchNormalization()
            self.use_bias = False
        if (activation != 'none'):
            if(activation=="leaky_relu"):
                self.activation = tf.nn.leaky_relu
            elif(activation=='relu'):
                self.activation = tf.nn.relu
            elif(activation=='elu'):
                self.activation = tf.nn.elu
            else:
                self.activation = tf.nn.relu

        self.conv = layers.Conv2D(output_dim, kernel_size, strides=stride, use_bias=self.use_bias, padding="same")

    def call(self, x):
        out = self.conv(x)
        if (hasattr(self, "norm")):
            out = self.norm(out)
        if (hasattr(self, "activation")):
            out = self.activation(out)
        return out

"""
    Sequential models
"""
class ResBlock(layers.Layer):

    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = models.Sequential(model)

    def call(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(layers.Layer):

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = models.Sequential(self.model)

    def call(self, x):
        return self.model(x)

"""
    VGG16
"""
class Vgg16(layers.Layer):

    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")
        self.conv1_2 = layers.Conv2D(64, kernel_size=3, stride=1, padding="same")

        self.conv2_1 = layers.Conv2D(128, kernel_size=3, stride=1, padding="same")
        self.conv2_2 = layers.Conv2D(128, kernel_size=3, stride=1, padding="same")

        self.conv3_1 = layers.Conv2D(256, kernel_size=3, stride=1, padding="same")
        self.conv3_2 = layers.Conv2D(256, kernel_size=3, stride=1, padding="same")
        self.conv3_3 = layers.Conv2D(256, kernel_size=3, stride=1, padding="same")

        self.conv4_1 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")
        self.conv4_2 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")
        self.conv4_3 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")

        self.conv5_1 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")
        self.conv5_2 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")
        self.conv5_3 = layers.Conv2D(512, kernel_size=3, stride=1, padding="same")

    def forward(self, X):
        h = tf.nn.relu(self.conv1_1(X))
        h = tf.nn.relu(self.conv1_2(h))
        # relu1_2 = h
        h = tf.nn.max_pool2d(h, kernel_size=2, stride=2)

        h = tf.nn.relu(self.conv2_1(h))
        h = tf.nn.relu(self.conv2_2(h))
        # relu2_2 = h
        h = tf.nn.max_pool2d(h, kernel_size=2, stride=2)

        h = tf.nn.relu(self.conv3_1(h))
        h = tf.nn.relu(self.conv3_2(h))
        h = tf.nn.relu(self.conv3_3(h))
        # relu3_3 = h
        h = tf.nn.max_pool2d(h, kernel_size=2, stride=2)

        h = tf.nn.relu(self.conv4_1(h))
        h = tf.nn.relu(self.conv4_2(h))
        h = tf.nn.relu(self.conv4_3(h))
        # relu4_3 = h

        h = tf.nn.relu(self.conv5_1(h))
        h = tf.nn.relu(self.conv5_2(h))
        h = tf.nn.relu(self.conv5_3(h))
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

# Discriminator
class MsImageDis(models.Model):

    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = layers.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.cnns = []

        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, padding=1, norm='none', activation=self.activ,
                              pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [
                Conv2dBlock(dim, dim * 2, 4, 2, padding=1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [layers.Conv2D(1, kernel_size=1, strides=1)]
        cnn_x = models.Sequential(cnn_x)
        return cnn_x

    def call(self, x):

        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)

        return outputs

    def calc_dis_loss(self, input_fake, input_real):

        # calculate the loss to train D
        outs0 = self.call(input_fake)
        outs1 = self.call(input_real)
        loss = 0.

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):

            if(self.gan_type=='lsgan'):
                loss += tf.reduce_mean((out0 - 0)**2) + tf.reduce_mean((out1 - 1)**2)
            elif(self.gan_type=='nsgan'):
                all0 = tf.Variable(tf.zeros_like(out0))
                all1 = tf.Variable(tf.ones_like(out1))
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out0,all0) + tf.nn.sigmoid_cross_entropy_with_logits(out0,all1))

            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        return loss

    def calc_gen_loss(self, input_fake):# calculate the loss to train G
        outs0 = self.call(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if(self.gan_type == 'lsgan'):
                loss += tf.reduce_mean((out0 - 1)**2) # LSGAN
            elif(self.gan_type=='nsgan'):
                all1 = tf.Variable(tf.ones_like(out0))
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out0, all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

# Generator
class AdaINGen(layers.Layer):

    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()

        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ,
                           pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def call(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # TODO
    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

class VAEGen(layers.Layer):

    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ,
                           pad_type=pad_type)

    def call(self, images):
        hiddens, noise = self.encode(images)
        if (self.training == True):
            # noise = tf.Variable(tf.random.normal(hiddens.shape))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = tf.Variable(tf.random.normal(hiddens.shape))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


"""
    Encoder and Decoders
"""

class StyleEncoder(layers.Layer):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []

        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [layers.GlobalAveragePooling2D()]  # 通道处理一下
        self.model += [layers.Reshape((1, 1, dim))]
        self.model += [layers.Conv2D(style_dim, kernel_size=1, strides=1)]
        self.model = models.Sequential(self.model)
        self.out_dim = dim

    def call(self, x):
        return self.model(x)


class ContentEncoder(layers.Layer):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # Conv()
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = models.Sequential(self.model)
        self.output_dim = dim

    def call(self, x):
        return self.model(x)


class Decoder(layers.Layer):

    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adam', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []

        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_upsample):
            self.model += [layers.UpSampling2D(),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='sigmoid', pad_type=pad_type)] # 傻逼吧, 解码不用sigmoid怎么映射到0-1
        self.model = models.Sequential(self.model)

    def call(self, x):

        return self.model(x)


class VAEGen(models.Model):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        self.training = True

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, norm='in', activ=activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ,
                           pad_type=pad_type)
        # self.noise = tf.random.normal((6,64,64,256))

    def call(self, images):
        hiddens, noise = self.encode(images)
        if (self.training == True):
            # noise = tf.Variable(tf.random.normal(hiddens.shape))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = tf.random.normal(hiddens.shape)
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


