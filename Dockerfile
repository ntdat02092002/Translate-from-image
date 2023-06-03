FROM python:3.9-slim
RUN mkdir /Translate-from-image
WORKDIR /Translate-from-image
COPY ./requirements.txt /Translate-from-image
RUN pip install --no-cache-dir -r /Translate-from-image/requirements.txt
RUN gdown --folder "https://drive.google.com/drive/folders/1tKHPwNgYsGusiR5C5VAYeli8g2C_jO3l?usp=sharing"
ADD . /Translate-from-image
CMD ["uvicorn", "my_api:app", "--host", "0.0.0.0", "--port", "80"]

