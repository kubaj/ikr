# Face and speaker recognition for IKR

## Architecture
We're using neural networks for classifying faces and speakers. Project is implemented using framework Keras with TensorFlow backend.

### Face recognition
Each face is converted to grayscale. Sobel filter is applied on the grayscale image.
Image has three layers:
 * grayscale image
 * image with sobel filter
 * zeros (Needed for image augmentation)

Image augmentation is used to generate new images using rotation.
Neural network consists of these layers:
 * 2D MaxPooling with kernel size 2x2
 * 2D Convolution with depth of 28, ReLu activation and kernel size 4x4
 * 2D MaxPooling with default values
 * Hidden layer of the size 512 with ReLu activation units
 * Output layer of the size 31 (number of classes)

There also dropout layers as prevention to the overfitting. Used optimizer is Adam with default values

### Speaker recognition

Each recording is dividied into windows with length of 25.6ms and 10ms step. Then 20 MFCC coefficients and 26 fbank coefficients are calculated for each time window.

Project consists of few parts: 
 * ikr-data - module for loading training and test data
 * ikr-face - script for training face recognition network
 * ikr-speech - script for training speaker recognition network
 * speechmap-eval - helper script for getting test accuracy of trained speech network model
 * infer - script for evaluating of the evaluation data
 * migrate-data - script for moving 1/2 of the test data to training data

## Installation

Python 3.6 and packages specified in `requirements.txt` are required for training and running the classifier.

The recommended way to install dependencies is to use [virtualenv](https://virtualenv.pypa.io/en/stable/) and pip:

```
$ virtualenv env
$ . env/bin/activate
(env)$ pip3 install -r requirements.txt
```

## Usage

### Training

First, it's necessary to train the networks and save the resulting models.

To start training of the face recognition network, launch:

```
$ python3 ikr-face.py
```

To start training of the speaker recognition network, launch:

```
$ python3 ikr-speech.py
```

Both scripts will ask whether you want to save the trained model after training.

Due to the way the speech recognition works, accuracy estimates displayed during training do not reflect real capabilities of the trained network. You can obtain a real estimate by running:

```
$ python3 speechmap-eval.py speechmap-model.h5
```

### Inference

After the training is complete, you can try to infer evaluation data using `infer.py`:

```
$ python3 infer.py path/to/eval/data/ --model-speech speechmap-model.h5 --model-face tenecaf-model.h5 -o results.txt
```

The results will be saved to file specified by the `-o` (`--output`) argument (in this case `results.txt`).

It is also possible to use only one of the models:

```
$ python3 infer.py path/to/eval/data/ --model-speech speechmap-model.h5 -o results-speech.txt
```

More details about supported arguments can be found in the help (`python3 infer.py --help`)
