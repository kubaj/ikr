import glob
import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.ndimage
from PIL import Image
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


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def open_image(filename):
    image = Image.open(filename).convert('L')
    image = np.array(image, np.float32)
    image, _ = image_histogram_equalization(image)

    # Edge detection
    sx = scipy.ndimage.sobel(image, axis=0, mode='constant')
    sy = scipy.ndimage.sobel(image, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    return np.array([sob, image, np.zeros(image.shape)])


def get_image_set(directory):
    filenames_l, labels_l = get_filenames_extension(directory, GLOB_FACES)

    images = []
    for img in filenames_l:
        image = open_image(img)
        images.append(image)

    images = np.array(images)
    labels = np.array(labels_l)

    return images, labels


def load_graphic_data():
    x_train, y_train = get_image_set(TRAIN_DIR)
    x_test, y_test = get_image_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)


def open_audio(filename):
    sample_rate, data = wav.read(filename)
    data = data[(2 * sample_rate):]  # cut off first 2 sec

    winlen = 0.0256
    winstep = 0.01

    # generate MFCC and fbanks
    coefficients = psf.mfcc(data, samplerate=sample_rate, numcep=20, winlen=winlen, winstep=winstep)
    coefficients_f = psf.logfbank(data, samplerate=sample_rate, winlen=winlen, winstep=winstep)

    # create maps from MFCC and fbanks
    mapheight = 15
    maps = []
    for i in range(0, len(coefficients), mapheight):
        if i + mapheight <= len(coefficients):
            ceff = coefficients[i:i + mapheight]
            ceff_f = coefficients_f[i:i + mapheight]
            maps.append(np.concatenate([ceff, ceff_f], axis=1))

    return maps


def get_speech_set(directory, pack=False):
    filenames_l, labels_l = get_filenames_extension(directory, GLOB_SPEECH)

    speeches = []
    labels = []
    for speech, lbl in zip(filenames_l, labels_l):
        maps = open_audio(speech)

        if not pack:
            speeches.extend(maps)
            labels.extend([lbl]*len(maps))
        else:
            speeches.append(np.array(maps))
            labels.append(lbl)

    speeches = np.array(speeches)
    labels = np.array(labels)

    return speeches, labels


def load_audio_data():
    x_train, y_train = get_speech_set(TRAIN_DIR)
    x_test, y_test = get_speech_set(DEV_DIR)

    return (x_train, y_train),(x_test, y_test)
