
"""
    @Author: sfwyly
    @Date: 2021/3/28
    @Description: process Dataset
"""

import numpy as np
import pathlib
import json
import platform

"""

    Dataset: provide more methods for operate Dataset

"""
class Dataset(object):

    """
        is_split: decide to whether split dataset
        image_size: image size
        root_path: data root path (including train or test)

    """
    def __init__(self, is_train = True, **kwargs):

        super(Dataset,self).__init__()
        self.trains = []
        self.labels = []
        self.isTrain = is_train
        self.category_to_label = {} # category2label
        self.labelNums = 0 # labels numbers
        self.all_file_path = [] # storage two-level category
        if("balanced" in kwargs):
            self.balanced = kwargs['balanced']

            if(self.balanced):
                self.balanced_num = kwargs['balanced_num']
        else:
            self.balanced = False

        self.root = kwargs['root_path']
        print("balanced ",self.balanced)
        sys = platform.system()
        if(sys=="Windows"):
            self.separator = "\\"
        else:
            self.separator = "/"
    def getAllFile(self):

        self.all_file_path.clear()
        file_root = pathlib.Path(self.root)
        self.all_file_path = list(file_root.glob("*/*"))#+list(file_root.glob("*/*.jpg"))
    """
        catagory_str : catagory path string
        enable to override
    """
    def processLabelStr(self,category_str) -> int:
        """
        :param category_str:
        :return: int
        """
        # TODO
        return len(self.category_to_label)

    def processLabelCategories(self,category_str):

        # here adding custom method to process catagory_to_label map
        label = self.processLabelStr(category_str)

        if(category_str not in self.category_to_label):

            self.category_to_label[category_str.strip()] = label
            self.labelNums += 1
        else:
            label = self.category_to_label[category_str.strip()]

        return label

    def __getitem__(self,item):

        """
        :param item:
        :return: only return path (train,label)
        """
        return self.trains[item],self.labels[item]

    def __len__(self):
        return len(self.trains)

    def shuffle(self):

        trainX = list(zip(self.trains, self.labels))
        np.random.shuffle(trainX)
        self.trains, self.labels = zip(*trainX)

    def loader(self, shuffle=True):
        self.trains.clear()
        self.labels.clear()

        if(len(self.all_file_path) <=0):
            if(self.balanced):
                self.balanceSample(self.balanced_num)
            else:
                self.getAllFile()

        for file_path in self.all_file_path:
            r = str(file_path).rfind(self.separator)
            l = str(file_path)[:r].rfind(self.separator)
            # label_content str(file_path)[l+1:r]

            self.labels.append(self.processLabelCategories(str(file_path)[l+1:r]))
            self.trains.append(str(file_path))
        if (shuffle):
            self.shuffle()
        return self.trains,self.labels

    def balanceSample(self, num=200):
        """
        :param num: sample num
        :return:
        """
        # balance sample
        self.all_file_path.clear()
        file_root = pathlib.Path(self.root).glob("*")
        for cate in file_root:
            img_list = list(cate.glob("*"))

            if (len(img_list) <= 0):
                continue
            if (len(img_list) >= num):

                self.all_file_path.extend(np.random.choice(img_list, num))
            else:
                self.all_file_path.extend(img_list)

    def setData(self,trains,labels):
        self.trains = trains[:]
        self.labels = labels[:]

    def setAllFilePaths(self,all_file_paths):
        self.all_file_path = all_file_paths[:]

    @staticmethod
    def splitTrainAndTest(root_path,split_ratio = 0.8,save_path = "/usr/",isWrite = False):

        """
            split Train and Val
            :return:
        """
        train_file_paths = []
        val_file_paths = []
        root = pathlib.Path(root_path)
        cate_list = root.glob("*")

        # recurrent each a cate
        for cate in cate_list:

            file_list = list(map(str,list(cate.glob("*"))))
            # no video sequence
            np.random.shuffle(file_list)
            length = len(file_list)
            train_file_paths.extend(file_list[:int(split_ratio*length)])
            val_file_paths.extend(file_list[int(split_ratio*length):])
        # write file
        if(isWrite):
            dic = {"train_file_paths":train_file_paths,"val_file_paths":val_file_paths}
            with open(save_path,"w") as file:
                json.dump(dic,file)
        return train_file_paths,val_file_paths

