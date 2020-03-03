# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
from numpy import linalg as LA

import keras
from keras.applications.resnet_v2 import resnet_v2
from keras.applications.resnet_v2 import preprocess_input
# from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import tensorflow as tf
import keras_applications
from RoiPooling import  RoiPooling
# #不加这几行就报错
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)



class ResNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (600, 600, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'

        #include_top = False表示只需要卷积层，但是我这里需要测试全连接所以include为true
        self.model = keras_applications.resnet.ResNet101(include_top=False,
                                                  weights=self.weight,
                                                  input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),pooling = self.pooling,
                                                  backend=keras.backend,
                                                  layers=keras.layers,
                                                  models=keras.models,
                                                  utils=keras.utils
                                                  )

        # self.basemodel = VGG16(weights=self.weight,)

        self.model.predict(np.zeros((1, 600, 600, 3)))
    
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        # print("------"+str(feat.shape[0])+"-----"+str(feat.shape[1]))
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        #改变输入的大小
        self.input_shape = (600, 600, 3)
        # self.input_shape = (None, None, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'

        #include_top = False表示只需要卷积层，但是我这里需要测试全连接所以include为true
        self.model = VGG19(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        # self.basemodel = VGG16(weights=self.weight,)
        #特征抽取可以改变全连接层，暂时没用
        # self.model = Model(input=self.basemodel.input,
        #                    outputs=self.basemodel.get_layer('fc1').output)
        # self.model.predict(np.zeros((1, 600, 600, 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        print(type(feat))

        print(feat.shape)
        # print("------"+str(feat.shape[0])+"-----"+str(feat.shape[1]))
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

