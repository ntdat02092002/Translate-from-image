import logging
import time
import os
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
from transformers import AutoModel, AutoTokenizer, BertTokenizer


MAX_TOKENS = 128
class TranslatorBaseTF(tf.Module):
  def __init__(self, tokenizer_en, tokenizer_vi, transformer):
    self.tokenizer_en = tokenizer_en
    self.tokenizer_vi = tokenizer_vi
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is English, hence adding the `[START]` and `[END]` tokens.
    # sentence = np.array([sentence])
    sentence = tf.convert_to_tensor(sentence)

    assert isinstance(sentence, tf.Tensor)

    # if len(sentence.shape) == 0:
    #   sentence = sentence[tf.newaxis]
    # print(sentence)
    # sentence = tf.convert_to_tensor(sentence)

    sentence = self.tokenizer_en.encode(sentence.numpy().decode("utf-8"))
    sentence = np.array(sentence)
    sentence = tf.convert_to_tensor(sentence)
    # print(sentence)
    encoder_input = sentence
    encoder_input = tf.expand_dims(encoder_input, 0)

    # As the output language is VietNamese, initialize the output with the
    # VietNamese `[START]` token.
    # start_end = self.tokenizer_vi.tokenize([''])[0]
    # start_end = self.tokenizer_vi.build_inputs_with_special_tokens([])
    # start = start_end[0][tf.newaxis]
    # end = start_end[1][tf.newaxis]
    start = tf.constant([self.tokenizer_vi.cls_token_id], dtype=tf.int64)
    end = tf.constant([self.tokenizer_vi.sep_token_id], dtype=tf.int64)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)


    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)


      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    # text = tokenizer_vi.detokenize(output)[0]  # Shape: `()`.
    # tokens = tokenizer_vi.lookup(output)[0])

    text = self.tokenizer_vi.decode(output.numpy()[0], skip_special_tokens=True)  # Shape: `()`.

    tokens = self.tokenizer_vi.convert_ids_to_tokens(output.numpy()[0])

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights




