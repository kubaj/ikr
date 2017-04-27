import os, glob, io
from PIL import Image
import numpy as np
from random import shuffle
import scipy.io.wavfile as wav
import python_speech_features as psf

GLOB_FACES = '*.png'
GLOB_SPEECH = '*.wav'
BASE_DIR = 'data/'
TRAIN_DIR = BASE_DIR + 'train/'
DEV_DIR = BASE_DIR + 'dev/'

def get_filenames_extension(directory, extension):
    filenames_l, labels_l = [], []
    for i in range(1, 32):
        filenames_new = [f for f in glob.glob('{}{}/{}'.format(directory, i, extension)) if os.path.isfile(f)]
        labels_l += [i-1 for _ in filenames_new]
        filenames_l += filenames_new
    return filenames_l, labels_l

def get_image_set(directory, shuff=False):
    filenames_l, labels_l = get_filenames_extension(directory, GLOB_FACES)

    if shuff:
        data = list(zip(filenames_l, labels_l))
        shuffle(data)
        filenames_l, labels_l = zip(*data)

    images = []
    for img in filenames_l:
        image = Image.open(img).convert('L')
        images.append(np.array(image))

    images = np.array(images)
    labels = np.array(labels_l)

    return images, labels

def load_graphic_data():

    x_train, y_train = get_image_set(TRAIN_DIR, shuff=True)
    x_test, y_test = get_image_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)

def get_speech_set(directory, shuff=False):
    filenames_l, labels_l = get_filenames_extension(directory, GLOB_SPEECH)

    if shuff:
        data = list(zip(filenames_l, labels_l))
        shuffle(data)
        filenames_l, labels_l = zip(*data)

    speeches = []
    labels = []
    for speech, lbl in zip(filenames_l, labels_l):
        # Preprocessing here? (removing empty parts)
        sample_rate, data = wav.read(speech)
        data = data[(2*sample_rate):]   # cut off first 2 sec

        winlen = 0.03
        winstep = 0.01

        # generate mfcc
        coefficients = psf.mfcc(data, samplerate=sample_rate, numcep=26, winlen=winlen, winstep=winstep)
        coefficients_f = psf.logfbank(data, samplerate=sample_rate, winlen=winlen, winstep=winstep)

        # create 26x13 maps of mfcc for convolution
        mapheight = 26
        for i in range(0, len(coefficients), mapheight):
            if i+mapheight <= len(coefficients):
                ceff = coefficients[i:i+mapheight]
                ceff_f = coefficients_f[i:i+mapheight]
                speeches.append(np.concatenate((ceff, ceff_f), axis=0))
                labels.append(lbl)

    speeches = np.array(speeches)
    labels = np.array(labels)

    return speeches, labels

def load_audio_data():
    x_train, y_train = get_speech_set(TRAIN_DIR)
    x_test, y_test = get_speech_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)
