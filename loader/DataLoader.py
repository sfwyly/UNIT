
"""
    @Author: sfwyly
    @Date: 2021/3/28
    @Description: process Dataset
"""

import cv2
from PIL import Image
from .Dataset import *
import functools


class DataLoader():

    def __init__(self, dataset: Dataset, batch_size: int, image_size=(256,256), shuffle: bool = True, is_mask = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.is_video = False
        self.is_mask = is_mask
        if(self.is_video):
            self.video_frames = 16 #get video frames
        # build our dataset
        self.build()

    def __len__(self):
        return len(self.trains)

    def __getitem__(self, item):
        return self.getBatchImage(self.trains[item],self.labels[item])
    """
        making dataset have batch
    """
    def shuffleAAndB(self,a,b):

        trainX = list(zip(a,b))
        np.random.shuffle(trainX)
        a,b = zip(*trainX)
        return a,b
    """
        build our dataset
    """
    def build(self):
        file_length = len(self.dataset)
        if(file_length<=0):
            self.dataset.loader() # loader dataset: train -> label
            file_length = len(self.dataset)
        # items
        items = int(np.ceil(file_length//self.batch_size))
        X_trains = []
        X_labels = []
        for i in range(items):
            image_list,labels = self.dataset.trains[i*self.batch_size:(i+1)*self.batch_size],self.dataset.labels[i*self.batch_size:(i+1)*self.batch_size]
            X_trains.append(image_list)
            X_labels.append(labels)

        if (self.shuffle):
            X_trains, X_labels = self.shuffleAAndB(X_trains, X_labels)

        self.trains = np.array(X_trains)
        self.labels = np.array(X_labels)

    # def train(self, model, valid_per_epochs=10, batch_size=8, epochs=100):
    #
    #     for epoch in range(1, epochs + 1):
    #         print("epoch: ", epoch, " / ", epochs)
    #         self.balanceSample()
    #         par = tqdm(range(int(np.ceil(len(self.X_trains) / batch_size))))
    #         val_loss, val_acc = 0, 0
    #         for i in par:
    #             X, Y = self.getBatchImage(self.X_trains[i * batch_size:(i + 1) * batch_size],
    #                                       self.X_labels[i * batch_size:(i + 1) * batch_size])
    #             # Y = np.array(self.X_labels[i*batch_size:(i+1)*batch_size]) - 1 # 分类 1 - 130 生成的时候转过来 #这里X 与 Y必须同batch 因为会过滤某些X
    #             Y_cate = to_categorical(Y, 130)
    #
    #             output = model.train_on_batch(X, Y_cate)
    #             # print(np.array(y_pred).shape)
    #             # print(y_pred)
    #             # loss = np.mean(tf.keras.losses.categorical_crossentropy(y_pred,Y_cate).numpy())
    #             # acc = sum((np.array(Y)-np.argmax(y_pred,-1))==0)/len(Y)
    #
    #             # print("loss: ",loss," acc: ",acc)
    #
    #             if (i % 100 == 0):
    #                 self.balanceSample()
    #                 #model.save_weights("/root/sfwy/jitu/ws.h5")
    #                 # 验证
    #                 val_loss, val_acc = self.valid(model, batch_size)
    #             par.set_description(
    #                 "loss: %.2f acc: %.2f val_loss: %.2f val_acc: %.2f" % (output[0], output[1], val_loss, val_acc))
    #         par.close()
    #         if (epochs % valid_per_epochs == 0):
    #             # 验证集.
    #             pass
    #
    # def valid(self, model, batch_size):
    #     loss = []
    #     acc = []
    #     self.shuffle(flag="valid")
    #     for i in range(10):
    #         # print("执行")
    #         X, Y = self.getBatchImage(self.val_trains[i * batch_size:(i + 1) * batch_size],
    #                                   self.val_labels[i * batch_size:(i + 1) * batch_size])
    #
    #         # print(X.shape,Y.shape)
    #         y_pred = model(X).numpy()
    #         # print("执行")
    #         Y_cate = to_categorical(Y, 130)
    #         los = np.mean(tf.keras.losses.categorical_crossentropy(y_pred, Y_cate).numpy())
    #         ac = sum((np.array(Y) - np.argmax(y_pred, -1)) == 0) / len(Y)
    #         loss.append(los)
    #         acc.append(ac)
    #     return np.mean(loss), np.mean(acc)
    def normalize(self,image,mode = 1):
        """
        normalize image
        :return:
        """
        if(mode == 1):
            image = image/255.
            image = np.clip(image, 0., 1.)
        elif(mode == 2):
            mean = []
            std = []
            image = (image - mean)/std
        return image

    def getBatchImage(self, file_path_list, y_labels):
        """
        get batch images from file path list
        :param file_path_list:
        :param y_labels:
        :return:
        """
        image_list = []
        labels = []
        for file_path, label in zip(file_path_list, y_labels):
            image = self.getImagesFromPath(file_path)
            if(self.is_video and len(image)==0):
                continue
            elif(self.is_video):
                new_image = []
                for img in image:
                    img = self.imageAugement(img, train=False)
                    new_image.append(img)
                image = np.array(new_image)
            else: # image
                if(self.is_mask):
                    h,w = image.shape

                    s_h = np.random.randint(0,h-self.image_size[0])
                    s_w = np.random.randint(0,w - self.image_size[1])

                    image = image[s_h:s_h + self.image_size[0], s_w:s_w + self.image_size[1]]
                else: # flip , rotate , movition 传numpy进去
                    image = self.imageAugement(image)

            # image = image.resize((self.image_size),Image.BILINEAR)
            image = self.normalize(image, mode=1)
            if ((image.shape)[-1] == 3 or self.is_mask):
                image_list.append(image)
                labels.append(label - 1)
        return np.array(image_list), np.array(labels)
    def rankVideoSequence(self, video_sequence):
        """
        rank video sequence
        :video_sequence is a video frame path sequence
        :return:
        """
        return video_sequence

    def getImagesFromPath(self, file_path):
        """
        get images from path: file_path is like a image path or a video (images sequence) category
        for example: /root/label/x.jpg or /root/label/video/x1-xn.jpg
        :param is_video:
        :return:
        """
        if(not self.is_video):
            image = Image.open(str(file_path))
            image = np.array(image)
            return image
        # deal video sequence for each file path
        video_squence_path = list(pathlib.Path(file_path).glob("*"))
        # each video frames need to rather than self.video_frames
        if(len(video_squence_path)<=self.video_frames):
            return []
        # video_squence = sorted(video_squence_path,)
        # customize self rank strategy for video is sequence
        video_squence_path = self.rankVideoSequence(video_squence_path)
        # obtain start position
        length = len(video_squence_path)
        start = np.random.randint(length-self.video_frames)
        video_squence = []
        for single_image_path in video_squence_path[start:start+self.video_frames]:
            image = Image.open(str(single_image_path))
            image = np.array(image)
            video_squence.append(image)
        return video_squence

    def imageAugement(self, image, mode="center", train = True):
        """
        image Augement
        :param image:
        :param mode:
        :param train:
        :return:
        """
        h, w, c = image.shape
        if(train):
            mode = "center" if np.random.randint(2) == 1 else "random"
        # noise = True if np.random.randint(2)==1 else False
        noise = False  # 添加噪声导致精确度不够
        # 先截取中间部位做 随机裁剪
        if (mode == "center"):
            if (h < w):
                offset = int((w - h) / 2)
                image = image[:, offset:h + offset, :]
            else:
                offset = int((h - w) / 2)
                image = image[offset:w + offset, :, :]
            image = Image.fromarray(np.uint8(image))
            image = np.array(image.resize((self.image_size), Image.BILINEAR))
        elif(mode == "scale"):
            image = Image.fromarray(np.uint8(image))
            if(h > w):
                new_size = (self.image_size[1], int(h/w * self.image_size[0]))
                image = np.array(image.resize(new_size, Image.BILINEAR))
                image = image[(new_size[1] - self.image_size[0])//2 + self.image_size[0],:,:]
            elif(w > h):
                new_size = (int(w / h * self.image_size[1]), self.image_size[0])
                image = np.array(image.resize(new_size, Image.BILINEAR))
                image = image[:, (new_size[0] - self.image_size[1]) // 2 + self.image_size[1], :]
            else:
                image = np.array(image.resize(self.image_size, Image.BILINEAR))
        else:  # random clip
            # h 434 w 401 均值近似
            image = Image.fromarray(np.uint8(image))
            image = np.array(image.resize((400, 430), Image.BILINEAR))
            s_h = np.random.randint(174)
            s_w = np.random.randint(144)
            image = image[s_h:s_h + 256, s_w:s_w + 256, :]

        # 是否添加高斯噪声
        # if (noise and (image.shape)[-1] == 3):
        #     sigma = np.random.randint(30)
        #     noise_image = np.random.randn(256, 256, 3) * sigma
        #     image = image + noise_image
        # 截取中间部分做水平或者垂直镜像
        # 是否进行镜像
        if(train):
            flag = np.random.randint(2)
            if(flag==1):
                image = cv2.flip(image,1)#np.random.randint(2))
        # 旋转会导致黑色底 需要想办法解决下
        # M = cv2.getRotationMatrix2D((sel.image_size//2,self.image_size/2),np.random.randint(90),np.random.randint(2))
        # image = cv2.warpAffine(image, M, self.image_size)
        return image
