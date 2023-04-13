import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import cv2
import os
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

class Opt():
    def __init__(self, saved_model, Transformation, FeatureExtraction, SequenceModeling, Prediction, sensitive=False):
        self.saved_model = saved_model
        self.Transformation = Transformation
        self.FeatureExtraction = FeatureExtraction
        self.SequenceModeling = SequenceModeling
        self.Prediction = Prediction

        #default arg
        self.workers = 4
        self.batch_size = 192
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive = sensitive
        self.PAD = False
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256

        if self.sensitive:
            self.character = string.printable[:-6]

        self.num_gpu = torch.cuda.device_count()


class Recogniter():
    def __init__(self, saved_model, Transformation, FeatureExtraction, SequenceModeling, Prediction, sensitive=False):
        self.opt = Opt(saved_model, Transformation, FeatureExtraction, SequenceModeling, Prediction, sensitive)

        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
              self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
              self.opt.SequenceModeling, self.opt.Prediction)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        self.model.load_state_dict(torch.load(self.opt.saved_model, map_location=self.device))

        self.model.eval()

    def infer(self, image_list):
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        # AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        # demo_data = RawDataset(root=image_folder, opt=self.opt)  # use RawDataset
        # demo_loader = torch.utils.data.DataLoader(
        #     demo_data, batch_size=self.opt.batch_size,
        #     shuffle=False,
        #     num_workers=int(self.opt.workers),
        #     collate_fn=AlignCollate_demo, pin_memory=True)
    
        # print(demo_loader)

        results = ""
        # ============= Preprocessing ==========
        list = []
        for img in image_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, [self.opt.imgW, self.opt.imgH])
            img = np.array(img) / 255.0
            list.append(img)

        list = np.array(list).reshape(len(list),1 , *list[0].shape)
        list = torch.from_numpy(list).float()
        list = list.unsqueeze(0)
        # ============= end pre =============

        batch_size = len(image_list)
        with torch.no_grad():
            for image_tensors in list:
                image = image_tensors.to(self.device)

                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds =self. model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)


                # log = open(f'./log_demo_result.txt', 'a')
                # dashed_line = '-' * 80
                # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

                # print(f'{dashed_line}\n{head}\n{dashed_line}')
                # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    results = results + pred + " "
                    # print(f'\t{pred:25s}\t{confidence_score:0.4f}')
                    # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

                # log.close()
        return results

if __name__ == '__main__':

    image_folder = 'demo_image'
    image_list = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_list.append(image)
    image_list = np.array(image_list)

    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()
    # demo(opt)
    recogniter = Recogniter("TPS-ResNet-BiLSTM-Attn.pth", "TPS", "ResNet", "BiLSTM", "Attn")
    ans = recogniter.infer(image_list) #path to folder contain word-croped images
    print(ans)
