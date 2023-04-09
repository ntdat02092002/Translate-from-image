from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import os
import fnmatch
import cv2
import string
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib

# char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# total number of our output classes: len(char_list)
# char_list = string.ascii_letters+string.digits

# # ============================= Model =====================
# # input with shape of height=32 and width=128
# inputs = Input(shape=(32,128,1))

# # convolution layer with kernel size (3,3)
# conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# # poolig layer with kernel size (2,2)
# pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

# conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
# pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

# conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)

# conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# # poolig layer with kernel size (2,1)
# pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

# conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# # Batch normalization layer
# batch_norm_5 = BatchNormalization()(conv_5)

# conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
# batch_norm_6 = BatchNormalization()(conv_6)
# pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

# conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)

# squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# # bidirectional LSTM layers with units=128
# blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
# blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)

# outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# # model to be used at test time
# act_model = Model(inputs, outputs)

# # load the saved best model weights
# act_model.load_weights('best_model.hdf5')

# img = cv2.cvtColor(cv2.imread("./TestData/4.png"), cv2.COLOR_BGR2GRAY)
# # convert each image of shape (32, 128, 1)
# img = cv2.resize(img, (128,32))
# img = np.expand_dims(img , axis = 2)
# # Normalize each image
# cc = []
# img = img/255.
# cc.append(img)
# cc = np.array(cc)

# # load the saved best model weights
# act_model.load_weights('best_model.hdf5')

# # predict outputs on validation images
# start_time = time.time()
# prediction = act_model.predict(cc)

# # use CTC decoder
# out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
#                          greedy=True)[0][0])

# # see the results
# i = 0
# for x in out:
#     # print("original_text =  ", valid_orig_txt[i])
#     print("predicted text = ", end = '')
#     for p in x:
#         if int(p) != -1:
#             print(char_list[int(p)], end = '')
#     print('\n')
#     i+=1
# end_time = time.time()
# print(end_time - start_time)


class OCR_CTC():
    def __init__(self,weights):
        self.__dict__.update(locals())

        # init model
        self.char_list = string.ascii_letters+string.digits
        inputs = Input(shape=(32,128,1))
        # convolution layer with kernel size (3,3)
        conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
        # poolig layer with kernel size (2,2)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
        conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
        conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
        conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
        # poolig layer with kernel size (2,1)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
        conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
        # Batch normalization layer
        batch_norm_5 = BatchNormalization()(conv_5)
        conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
        conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
        # bidirectional LSTM layers with units=128
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
        outputs = Dense(len(self.char_list)+1, activation = 'softmax')(blstm_2)
        # model to be used at test time
        self.model = Model(inputs, outputs)
        self.model.load_weights(weights)
    def pre_image(self, images):
        self.test_img = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # convert each image of shape (32, 128, 1)
            img = cv2.resize(img, (128,32))
            img = np.expand_dims(img , axis = 2)
            # Normalize each image
            img = img/255.
            self.test_img.append(img)
        self.test_img = np.array(self.test_img)
        return self.test_img
    def infer(self, test_image):
        result = ""
        # start_time = time.time()
        prediction = self.model.predict(self.pre_image(test_image))

        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                greedy=True)[0][0])

        # see the results
        i = 0
        for x in out:
            # print("predicted text = ", end = '')
            for p in x:
                # print(x)
                if int(p) != -1:
                    result = result + self.char_list[int(p)]
                    # print(self.char_list[int(p)], end = '')
            # print('\n')
            result = result + " "
            i+=1
        end_time = time.time()
        return result
        # print(end_time - start_time)

# model_path = "best_model.hdf5"
# ocr_ctc = OCR_CTC(weights=model_path)

# image = cv2.imread("./TestData/6.png")
# imgs = []
# imgs.append(image)
# kk = ocr_ctc.infer(imgs)
# print(kk)
