import CRAFT_pytorch
# from CRAFT_pytorch.my_CRAFT import My_CRAFT
from CRAFT_pytorch.odering_output import My_CRAFT
# import text_recognition
# from text_recognition.my_recogniter import Recogniter
# import translation
# from translation.my_translator import Translator

import numpy as np
from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

if __name__ == '__main__':
    img_path = "image/test7.png"

    img = loadImage(img_path)

    detector =  My_CRAFT(trained_model="models/craft_mlt_25k.pth")
    # recogniter = Recogniter("models/TPS-ResNet-BiLSTM-CTC.pth", "TPS", "ResNet", "BiLSTM", "CTC")
    # translator = Translator("models/TransEnVi.ckpt")
     #path to folder contain word-croped images

    # word_images = detector.detect(img)
    # results = recogniter.infer(word_images)
    # translated = translator.translate(results)
    # print(translated)

    image_list = detector.detect(img)
    print(image_list)
    for image_ in image_list:
        cv2.imshow("fds", image_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print(horizontal_list)
    # print(free_list)
