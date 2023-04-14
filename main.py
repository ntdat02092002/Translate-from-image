import CRAFT_pytorch
from CRAFT_pytorch.my_CRAFT import My_CRAFT
import text_recognition
from text_recognition.my_recogniter import Recogniter

if __name__ == '__main__':
    img_path = "image/cc.png"

    detector =  My_CRAFT(trained_model="craft_mlt_25k.pth")
    recogniter = Recogniter("TPS-ResNet-BiLSTM-CTC.pth", "TPS", "ResNet", "BiLSTM", "CTC")
     #path to folder contain word-croped images

    word_images = detector.detect(img_path)
    results = recogniter.infer(word_images)
    print(results)
