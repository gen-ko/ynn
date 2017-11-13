# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
import src.nlp as nlp
import matplotlib.pyplot as plt
import numpy
import os
import pickle
from src import util as uf
from src import callback as cb
from src import train as utf


# resolve file path
path, _ = os.path.split(os.path.realpath(__file__))
dictionary_filename = os.path.join(path,'../output/dump', 'dictionary.dump')

# load vocabulary
with open(dictionary_filename, 'rb') as f:
    d = pickle.load(f)

nn_filename = os.path.join(path, '../output/dump', 'script-3-2-h128-i-s.dump')
nn = nlp.NlpL3TypeA(8000, 16, 128)
with open(nn_filename, 'rb') as f:
    nn.load(pickle.load(f))

sentence = utf.nlp_inference_sentence(nn, ['government', 'of', 'united'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['city', 'of', 'new'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['life', 'in', 'the'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['he', 'is', 'the'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['there', 'are', 'millions'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['it', 'is', 'not'], d)
print(sentence)

sentence = utf.nlp_inference_sentence(nn, ['in', 'the', 'next'], d)
print(sentence)