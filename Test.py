# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:


Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np

from keras.models import Model


from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import json
import embedding

from keras.layers import Input, Dense, Embedding, GRU
from keras.utils.layer_utils import merge
#from keras.layers import merge

from keras import backend as K

# load models
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x)
    return np.expand_dims(features.flatten(), axis=0)


def load(captions_filename, features_filename):
    features = np.load(features_filename)
    images = []
    texts = []
    with open(captions_filename) as fp:
        for line in fp:
            tokens = line.strip().split()
            images.append(tokens[0])
            texts.append(' '.join(tokens[1:]))
    return features, images, texts

def custom_loss(y_true, y_pred):
    positive = y_pred[:,0]
    negative = y_pred[:,1]
    return K.sum(K.maximum(0., 1. - positive + negative))

def accuracy(y_true, y_pred):
    positive = y_pred[:,0]
    negative = y_pred[:,1]
    return K.mean(positive > negative)

def preprocess_texts(texts):
    output = []
    for text in texts:
        output.append([vocab[word] if word in vocab else 0 for word in text.split()])
    return pad_sequences(output, maxlen=16)


def generate_caption(image_filename, n=10):
    # generate image representation for new image
    image_representation = image_model.predict(extract_features(image_filename))
    # compute score of all captions in the dataset
    scores = np.dot(caption_representations,image_representation.T).flatten()
    # compute indices of n best captions
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]



    # display them
    for i in [int(x) for x in reversed(indices)]:
        print(scores[i], texts[i])

def search_image(caption, n=10):
    caption_representation =caption_model.predict(preprocess_texts([caption]))
    scores = np.dot(image_representations,caption_representation.T).flatten()
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]
    for i in [int(x) for x in reversed(indices)]:
        print(scores[i], images[i])

if __name__ == '__main__':
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    features, images, texts = load('annotations.10k.txt', 'resnet50-features.10k.npy')
    image_model = load_model('model.image')
    caption_model = load_model('model.caption')

    # load representations (you could as well recompute them)
    import numpy as np

    caption_representations = np.load('caption - representations.npy')
    image_representations = np.load('image - representations.npy')

    vocab = json.loads(open('vocab.json').read())

    # generate a caption for an image
    print('1: generate captions for an image')
    generate_caption('2.png')

    # search the image corresponding to this caption
    print(' ')
    print('2: search the image corresponding to this caption')
    search_image('a man in the snow on some skis')

