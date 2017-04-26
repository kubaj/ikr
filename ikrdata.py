import os, glob, io
from PIL import Image
import numpy as np
from random import shuffle
import scipy

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
    for speech in speeches:
        # Preprocessing here? (removing empty parts)
        sample_rate,data = scipy.io.wavfile.read(speech)
        speeches.append(data)

    speeches = np.array(speeches)
    labels = np.array(labels_l)

    return speeches, labels

def load_audio_data():
    x_train, y_train = get_speech_set(TRAIN_DIR, shuff=True)
    x_test, y_test = get_speech_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)