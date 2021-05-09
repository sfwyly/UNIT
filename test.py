


# 局部样式+局部对抗损失
import sys, os

import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from networks import *
from utils import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

configs = get_config("configs/unit_edges2handbags_folder.yaml")

gen_a = VAEGen(3,configs['gen'])
gen_a.build(input_shape=(configs['batch_size'],configs['image_size'],configs['image_size'],3))
gen_b = VAEGen(3,configs['gen'])
gen_b.build(input_shape=(configs['batch_size'],configs['image_size'],configs['image_size'],3))

gen_a.load_weights(configs['save_path']+"gen_a.h5")
gen_b.load_weights(configs['save_path']+"gen_b.h5")

def getAllImagePath(name):
    path_root = pathlib.Path(name)
    result = list(path_root.glob("*/*"))

    return np.array(result)

def getDataset(all_image_paths):
    train = []
    # labels = []
    for path in all_image_paths:
        path = str(path)
        # 添加图片
        image = Image.open(path)
        # image = image.resize((256, 256), Image.BILINEAR)  # *218//178
        image = np.array(image)
        h, w, c = image.shape
        if (True):
            if (h < w):
                offset = int((w - h) / 2)
                image = image[:, offset:h + offset, :]
            else:
                offset = int((h - w) / 2)
                image = image[offset:w + offset, :, :]
            image = Image.fromarray(np.uint8(image))
            image = np.array(image.resize((256,256), Image.BILINEAR))
        train.append(image)  # [24:280,...])
    return np.array(train)

def getPSNR(image, true_image):
    height, width, _ = image.shape
    channel_mse = np.sum((image - true_image) ** 2, axis=(0, 1)) / (height * width)
    mse = np.mean(channel_mse)
    Max = 1.0  # 最大图像
    PSNR = 10. * np.log10(Max ** 2 / mse)  # 峰值信噪比

    return PSNR


def l1_loss(y_pred, y_true, mask_list):
    # print("l1")
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    return 100. * tf.reduce_mean(tf.abs(y_pred - y_true)) + 6. * tf.reduce_mean(
        tf.abs(y_pred * (1 - mask_list) - y_true * (1 - mask_list)))


def getMaskListPaths(name):
    path_root = pathlib.Path(name)
    mask_paths = list(path_root.glob("*/*.png"))

    return np.array(mask_paths)

def getMaskList(mask_paths,image_size = (256,256)):
    mask_list = []
    for path in mask_paths:
        path = str(path)
        # 添加图片
        image = np.array(Image.open(path))
        h, w = image.shape

        s_h = np.random.randint(0, h - image_size[0])
        s_w = np.random.randint(0, w - image_size[1])

        image = image[s_h:s_h + image_size[0], s_w:s_w + image_size[1]][...,np.newaxis]
        mask_list.append(1. - image / 255.)
    return np.array(mask_list)

if (__name__ == "__main__"):
    all_image_paths = getAllImagePath(configs['train_path']) # data_256/z/zen_garden F:/place2/val_256/
    # np.random.shuffle(all_image_paths) paris_eval_gt paris_train_original F:/datasets/Paris_StreetView/Paris_StreetView_Dataset/paris_eval_gt/
    print(len(all_image_paths))

    dataset = getDataset(all_image_paths[24:34]) / 255.  # 73
    mask_paths = getMaskListPaths(configs['train_mask_path'])  # E:/cnn/comparsion/mask/ F:/mask/testing_mask_dataset/
    mask_list = getMaskList(mask_paths[9993:9994])  # [_ for _ in range(8000)],1)] np.random.choice(mask_paths,1)

    # mask_list = getHoles((256,256),1)[...,np.newaxis]
    x = dataset[:6] * mask_list

    i = 0
    images, noise = gen_a.encode(x[i:i + 6, ...])
    result = gen_b.decode(images+noise).numpy()
    plt.figure(1)
    plt.subplot(321)
    plt.imshow(x[i + 0])
    plt.subplot(322)
    plt.imshow(result[0])
    plt.subplot(323)
    plt.imshow(x[i + 1])
    plt.subplot(324)
    plt.imshow(result[1])
    plt.subplot(325)
    plt.imshow(x[i + 2])
    plt.subplot(326)
    plt.imshow(result[2])  # *0.5 +0.5

    plt.savefig(configs['save_path']+"result.png")
    plt.show()



