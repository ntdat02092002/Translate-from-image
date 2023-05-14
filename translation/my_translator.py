import logging
import time
import os
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from transformer import Transformer
from utils import CustomSchedule, print_translation
from translator import TranslatorBaseTF


BUFFER_SIZE = 20000
BATCH_SIZE = 64

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

class Translator():
    def __init__(self, checkpoint_path):
        self.tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_vi = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.transformer = Transformer(num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=self.tokenizer_en.vocab_size,
            target_vocab_size=self.tokenizer_vi.vocab_size,
            dropout_rate=dropout_rate)

        self.transformer.load_weights(checkpoint_path).expect_partial()

        self.translator = TranslatorBaseTF(self.tokenizer_en, self.tokenizer_vi, self.transformer)

    def translate(self, sentence):
        translated_text, translated_tokens, attention_weights = self.translator(sentence)
        # print_translation(sentence, translated_text, ground_truth)
        return translated_text


if __name__ == '__main__':

    checkpoint_path = "./models/TransEnVi.ckpt"

    sentence = 'How old are you ?'

    # print(tf.convert_to_tensor(sentence))

    translator = Translator(checkpoint_path)

    result = translator.translate(sentence)

    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {result}')
