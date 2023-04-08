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
import craft_utils
import test
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

# def str2bool(v):
#     return v.lower() in ("yes", "y", "true", "t", "1")

def sort_words(data, words, y_threshold):
    data = np.array(data)
    # Sắp xếp các hộp chứa theo giá trị tọa độ y
    sorted_data = data[data[:, 1].argsort()]

    # Khởi tạo danh sách để lưu trữ kết quả
    result = []

    # Duyệt qua các hộp chứa đã sắp xếp
    current_row = []
    for i in range(sorted_data.shape[0]):
        # Nếu hộp chứa hiện tại thuộc cùng hàng với hộp chứa trước đó
        if i > 0 and abs(sorted_data[i, 1] - sorted_data[i-1, 1]) <= y_threshold:
            current_row.append(sorted_data[i])
        else:
            # Nếu current_row không rỗng
            if current_row:
                # Tính toán giá trị trung bình của các giá trị y trong current_row
                mean_y = np.mean(current_row, axis=0)[1]
                # Gán giá trị trung bình cho tất cả các hộp chứa trong current_row
                current_row = np.array(current_row)
                current_row[:, 1] = mean_y
                # Sắp xếp các hộp chứa trong current_row theo giá trị tọa độ x
                current_row = current_row[current_row[:, 0].argsort()]
                result.append(current_row)
            current_row = [sorted_data[i]]

    # Thêm hàng cuối cùng vào kết quả
    if current_row:
        mean_y = np.mean(current_row, axis=0)[1]
        current_row = np.array(current_row)
        current_row[:, 1] = mean_y
        current_row = current_row[current_row[:, 0].argsort()]
        result.append(current_row)

    new_sx = []
    new_words = []
    for i in result:
        for j in i:
            new_sx.append(j)
    for i in new_sx:
        new_words.append(words[i[2]])
    return new_words, result

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

# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

# args = parser.parse_args()

""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


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

    def detect(self, image_path):
        # list image cropped to show
        word_images = []
        sx = []

        t = time.time()

        # load data
        print("Test image {:s}".format(image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test.test_net(self.net, image, self.text_threshold, self.link_threshold,
            self.low_text, self.cuda, self.poly, self.canvas_size, self.mag_ratio, self.refine_net)

        dem = 0
        for box_num in range(len(bboxes)):
            pts = bboxes[box_num]
            if np.all(pts) > 0:
                word, idx = crop(pts, image)
                idx.append(dem)
                sx.append(idx)
                word_images.append(word)
                dem += 1

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

        print("elapsed time : {}s".format(time.time() - t))
        words, re = sort_words(sx, word_images, 15)
        return words

if __name__ == '__main__':
    img_path = "image/cc.png"
    weight_path = "craft_mlt_25k.pth"

    # list image cropped to show
    model = My_CRAFT(weight_path)
    word_images = model.detect(img_path)
    # print(len(word_images))
    word = word_images[3]
    cv2.imshow("word", word)
    cv2.waitKey(0)
    cv2.destroyAllWindows()