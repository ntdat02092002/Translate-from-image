import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from CRAFT_pytorch import craft_utils
from CRAFT_pytorch import test
from CRAFT_pytorch import imgproc
from CRAFT_pytorch import file_utils
import json
import zipfile

from CRAFT_pytorch.craft import CRAFT

from collections import OrderedDict

from CRAFT_pytorch.odering_output_utils import group_text_box, diff, get_image_list, crop_combine_list


def crop(pts, image):

    """
    Takes inputs as 8 points
    and Returns cropped, masked image with a white background
    """
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = image[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg + dst

    return dst2, [x, y]


class My_CRAFT():
    def __init__(self, trained_model, text_threshold=0.7, cuda=False):
        self.net = CRAFT()
        self.cuda = cuda
        self.text_threshold = text_threshold
        self.trained_model = trained_model

        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.refine = False
        self.refine_model = 'weights/craft_refiner_CTW1500.pth'
        self.poly = False
        self.low_text = 0.4
        self.link_threshold = 0.4

        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            self.net.load_state_dict(test.copyStateDict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(test.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if self.refine:
            from refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refine_model + ')')
            if self.cuda:
                self.refine_net.load_state_dict(test.copyStateDict(torch.load(self.refine_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(test.copyStateDict(torch.load(self.refine_model, map_location='cpu')))

            self.refine_net.eval()
            POLY = True

    def detect(self, image):
        # list image cropped to show
        word_images = []
        sx = []

        t = time.time()

        bboxes, polys, score_text = test.test_net(self.net, image, self.text_threshold, self.link_threshold,
            self.low_text, self.cuda, self.poly, self.canvas_size, self.mag_ratio, self.refine_net)

        print(len(bboxes))
        # dem = 0
        # for box_num in range(len(bboxes)):
        #     pts = bboxes[box_num]
        #     if np.all(pts) > 0:
        #         word, idx = crop(pts, image)
        #         idx.append(dem)
        #         sx.append(idx)
        #         word_images.append(word)
        #         dem += 1


        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)

        #odering bboxs
        combine_list, free_list = group_text_box(single_img_result) # use all default param


        # crop image
        image_list = crop_combine_list(combine_list, image)

        free_image_list = crop_free_list(free_list, image)
        image_list.extend(free_image_list)

        return image_list
