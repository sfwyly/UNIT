

import tensorflow as tf
import numpy as np

style_layers = ['block1_pool' ,'block2_pool' ,'block3_pool']
content_layers = []
# num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def high_pass_x_y(image):
    x_var = image[: ,: ,1: ,:] - image[: ,: ,:-1 ,:]
    y_var = image[: ,1: ,: ,:] - image[: ,:-1 ,: ,:]

    return x_var, y_var

# def total_variation_loss(image):
#  x_deltas, y_deltas = high_pass_x_y(image)
#  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)
def total_variation_loss(image ,mask_list):

    kernel = tf.ones((3 ,3 ,mask_list.shape[3] ,mask_list.shape[3]))
    dilated_mask = tf.nn.conv2d( 1 -mask_list ,kernel ,strides=[1 ,1 ,1 ,1] ,padding="SAME")
    dilated_ratio = 9. * 3 /(dilated_mask +10e-6)
    dilated_mask = tf.cast(tf.greater(dilated_mask ,0) ,"float32")
    image = dilated_mask * image
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG
    # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # vgg = tf.keras.models.load_model("/content/drive/My Drive/data/my_segmentation_model.h5")
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1 ] *input_shape[2 ] *input_shape[3], tf.float32)
    return result /(num_locations)

# def gram_matrix(x):
#     """Gram matrix used in the style losses."""
# #     assert K.ndim(x) == 4, 'Input tensor should be 4D (B, H, W, C).'
# #     assert K.image_data_format() == 'channels_last', "Use channels-last format."

#     # Permute channels and get resulting shape
#     x = tf.transpose(x, (0, 3, 1, 2))

#     shape = x.shape
#     B, C, H, W = shape[0], shape[1], shape[2], shape[3]

#     # Reshape x and do batch dot product
#     features = tf.reshape(x, K.stack([B, C, H*W]))

#     gram = K.batch_dot(features, features, axes=2)

#     # Normalize with channels, height and width
#     gram /= K.cast(C * H * W, x.dtype)

#     return gram

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):

        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        # self.vgg.summary()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        # inputs = tf.image.resize(inputs, (224, 224))
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers] ,outputs[self.num_style_layers:])
        # style层的原始输出
        perceptual_dict = {style_name :value
                           for style_name, value
                           in zip(self.style_layers, style_outputs)}
        style_dict = {style_name :gram_matrix(value)
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'perceptual' :perceptual_dict, 'style' :style_dict}

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=10e-8, clip_value_max=1.0 -10e-8)

# style_weight=1e-2
# content_weight=1e4

# content_image 其实就是图片显示区域，style_image是局部样式，必须保持content_image的强一致
def style_content_loss(outputs ,style_targets):
    # print("样式")
    style_outputs = outputs['style']
    style_targets = style_targets['style']

    # content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean(tf.abs(style_outputs[name ] -style_targets[name] )**2)
                           for name in style_outputs.keys()])
    return style_loss
def l1_loss(y_pred ,y_true ,mask_list):
    # print("l1")
    y_pred = tf.cast(y_pred ,dtype=tf.float32)
    y_true = tf.cast(y_true ,dtype=tf.float32)
    return 1. *tf.reduce_mean(tf.abs(y_pred - y_true) ) +5. *tf.reduce_mean(tf.abs(y_pred - y_true ) *( 1 -mask_list))

def cal_perceptual(outputs ,style_targets):
    # print("样")
    style_outputs = outputs['perceptual']
    style_targets = style_targets['perceptual']

    result = tf.add_n \
        ([tf.reduce_mean(tf.abs(style_outputs[name ] -style_targets[name])) for name in style_outputs.keys()])
    return result
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits= False)

# 这里暂时只计算生成器的对抗
def cal_adv(real_image_list ,fake_image_list):
    # print("对抗")
    real_loss = cross_entropy(tf.ones_like(real_image_list), clip_0_1(real_image_list))
    fake_loss = cross_entropy(tf.zeros_like(fake_image_list), clip_0_1(fake_image_list))
    total_loss = real_loss + fake_loss
    return total_loss

def cal_gen(fake_image_list):
    return cross_entropy(tf.ones_like(fake_image_list), clip_0_1(fake_image_list))

def resize_image(batch_image,in_size):

    B ,H ,W ,C = batch_image.shape
    insize = 224

    batch_image = (batch_image +1 ) * 127.5 - [123.68, 116.779 ,103.939]
    limx = H - in_size
    limy = W - in_size
    xs = np.random.randint(0 ,limx ,B)
    ys = np.random.randint(0 ,limy ,B)

    return np.array(batch_image)