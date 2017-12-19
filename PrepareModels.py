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


if __name__ == '__main__':

    #features = np.load('resnet50-features.10k.npy')
    #print(features.shape)

    resnet_model = ResNet50(weights='imagenet', include_top=False)

    #features = extract_features('2.png')
    #texts = []
    features, images, texts = load('annotations.10k.txt', 'resnet50-features.10k.npy')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    captions = pad_sequences(sequences, maxlen=16)

    vocab = tokenizer.word_index
    vocab['<eos>'] = 0 # add word with id 0

    with open('vocab1.json', 'w') as fp: # save the vocab
        fp.write(json.dumps(vocab))

    embedding_weights = embedding.load(vocab, 100, 'glove.twitter.27B.100d.filtered.txt')


    image_input = Input(shape=(2048,))
    caption_input = Input(shape=(16,))
    noise_input = Input(shape=(16,))

    caption_embedding = Embedding(len(vocab), 100, input_length=16, weights=[embedding_weights])
    caption_rnn = GRU(256)
    image_dense = Dense(256, activation='tanh')

    image_pipeline = image_dense(image_input)
    caption_pipeline = caption_rnn(caption_embedding(caption_input))
    noise_pipeline = caption_rnn(caption_embedding(noise_input))

    positive_pair = merge([image_pipeline, caption_pipeline], mode='dot')
    negative_pair = merge([image_pipeline, noise_pipeline], mode='dot')
    output = merge([positive_pair, negative_pair], mode='concat')

    training_model = Model(input=[image_input, caption_input,  noise_input], output=output)
    image_model = Model(input=image_input, output=image_pipeline)
    caption_model = Model(input=caption_input, output=caption_pipeline)


    training_model.compile(loss=custom_loss, optimizer='adam', metrics=[accuracy])

    noise = np.copy(captions)
    fake_labels = np.zeros((len(features), 1))
    X_train = [features[:9000], captions[:9000], noise[:9000]]
    Y_train = fake_labels[:9000]
    X_valid = [features[-1000:], captions[-1000:], noise[-1000:]]
    Y_valid = fake_labels[-1000:]

    # actual training
    for epoch in range(10):
        np.random.shuffle(noise) # don’t forget to shuffle mismatched captions
        training_model.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], nb_epoch=1, batch_size=64)

    # save models
    image_model.save('model.image1')
    caption_model.save('model.caption1')
    # save representations
    np.save('caption - representations1', caption_model.predict(captions))
    np.save('image - representations1', image_model.predict(features))



