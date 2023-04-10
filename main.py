import CRAFT_pytorch
from CRAFT_pytorch.my_CRAFT import My_CRAFT
import text_recognition
from text_recognition.demo_ocr_ctc import OCR_CTC

if __name__ == '__main__':
    img_path = "image/cc.png"

    detector =  My_CRAFT(trained_model="craft_mlt_25k.pth")
    recogniter = OCR_CTC(weights="best_model.hdf5")

    word_images = detector.detect(img_path)
    results = recogniter.infer(word_images)
    print(results)
