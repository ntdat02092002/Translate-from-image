# Translate-from-image

## Introduction
The problem of combining object detection and object recognition is a difficult problem, as both object detection and object recognition methods must be combined. That's why in this repo we want to challenge ourselves with an application that can both detect text and recognize detected text areas. In addition, to make the problem more challenging, we also use a machine translation model.

In this application we use [CRAFT](https://arxiv.org/abs/1904.01941) for text detection, [TRBA](https://openaccess.thecvf.com/content_ICCV_2019/html/Baek_What_Is_Wrong_With_Scene_Text_Recognition_Model_Comparisons_Dataset_ICCV_2019_paper.html) model for text recognition and [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) model for text translation from English to Vietnamese.

![this_repo_inout](https://github.com/TruongNoDame/Translate-text-from-image/blob/main/text_recognition/demo_image/Picture1.png)

## Run demo

#### To run this repo, follow these steps if you don't want to use docker:

```
git clone this repo
```
Before that, let create a folder named models in Translate-text-from-image, you can download all file in [this folder](https://drive.google.com/drive/u/2/folders/1mh3rzYTAUs-lcCuzqLlX8zMP4MyvMhzd).

```
cd Translate-text-from-image
pip install -r requirements.txt
python/python3 my_api.py
```

After running all code above, you can try it by your browser. Let put `http://localhost:8000/text` in your browser.


#### To run this repo use docker:
```
git clone this repo
cd Translate-text-from-image
docker build -t image_name
docker run -it --name container_name --network=host --ipc=host -p 8000:80 image_name
```
Where image_name and container_name are the name of the image and the name of the container you want to give them

## Usage

### Translate from text
To use the application, you can directly transmit the English text that needs to be translated into Vietnamese into the text field on the left, then press the "Translate" button, the translated text will return as Vietnamese on the right ( red area).
![use_input_text](https://github.com/TruongNoDame/Translate-text-from-image/blob/main/text_recognition/demo_image/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202023-12-28%20202104.png)

### Translate from image
To use this mode, you need to select a photo from your computer using the "Choose file" button. After the photo is uploaded, press the Translate button and wait for the returned result to be the text translated from the transferred image.
![use_input_image](https://github.com/TruongNoDame/Translate-text-from-image/blob/main/text_recognition/demo_image/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202023-12-28%20203400.png)
