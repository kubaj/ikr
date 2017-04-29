from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LocallyConnected2D
from keras import backend as K
import ikrdata

batch_size = 32
num_classes = 31
epochs = 200
img_rows, img_cols = 60, 60
input_channels = 3
modelfile = 'facenet'

(x_train, y_train), (x_test, y_test) = ikrdata.load_graphic_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], input_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], input_channels, img_rows, img_cols)
    input_shape = (input_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, input_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, input_channels)
    input_shape = (img_rows, img_cols, input_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(
    32,
    kernel_size=(4, 4),
    activation='relu',
    input_shape=input_shape,
    bias_initializer=initializers.Constant(0.1),
))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(
    64, (2, 2),
    activation='relu',
    bias_initializer=initializers.Constant(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(
    256,
    activation='relu',
    bias_initializer=initializers.Constant(0.1)))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if input('Save model? (y): ').lower() == 'y':
    path = '{}-acc{}.h5'.format(modelfile, score[1])
    model.save(path)
    print('Model saved to', path)
