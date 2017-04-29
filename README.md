# Face and speaker recognition for IKR

## Architecture
We're using neural networks for classifying faces and speakers. Project is implemented using framework Keras with Tensorflow backend.

### Face recognition
Each face is converted to grayscale. Sobel's filter is applied on the grayscale image.
Image has three layers:
 * grayscale image
 * image with sobel's filter
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
 * migrate-data - script for moving 1/2 of the test data to training datas

## Usage
Make sure you have training data in folder `data/`
Install dependencies (can be installed to virtualenv):
```
$ pip install -r requirements.txt
```
Run training of the face recognition network:
```
$ python ikr-face.py
```
Run training of the speaker recognition network:
```
$ python ikr-speech.py
```
You can get test accuracy of the network using:
```
$ python speechmap-eval.py *.h5
```



