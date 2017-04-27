import os, glob, io
from PIL import Image
import numpy as np
from random import shuffle
import scipy.io.wavfile as wav
from scipy.signal.filter_design import butter, buttord
from scipy.signal import lfilter
from python_speech_features import mfcc

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

def convert_hertz(freq):
    # convert frequency in hz to units of pi rad/sample
    # (our .WAV is sampled at 44.1KHz)
    return freq * 2.0 / 44100.0

def passband_filter(samples):
    pass_freq = convert_hertz(280.0)
    stop_freq = convert_hertz(3500.0)
    pass_gain = 3.0  # tolerable loss in passband (dB)
    stop_gain = 60.0  # required attenuation in stopband (dB)
    ord, wn = buttord(pass_freq, stop_freq, pass_gain, stop_gain)
    b, a = butter(ord, wn, btype='low')
    return lfilter(b, a, samples)

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
        start = int(2*sample_rate)
        start_noise = start - int(0.2*sample_rate)

        threshold = np.mean(np.abs(data[start_noise:start]))
        data = data[start:]   # cut off first 2 sec
        ndata = data[data > (threshold*1.0)]

        # generate mfcc
        coefficients = mfcc(ndata, samplerate=sample_rate, winlen=0.03, numcep=26)

        # create 26x13 maps of mfcc for convolution
        mapheight = 26
        for i in range(0, len(coefficients), mapheight):
            if i+mapheight <= len(coefficients):
                speeches.append(coefficients[i:i+mapheight])
                labels.append(lbl)

    speeches = np.array(speeches)
    labels = np.array(labels)

    return speeches, labels

def load_audio_data():
    x_train, y_train = get_speech_set(TRAIN_DIR)
    x_test, y_test = get_speech_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)
