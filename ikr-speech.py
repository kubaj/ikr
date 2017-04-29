from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.local import LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import ikrdata

batch_size = 600
num_classes = 31
epochs = 32
maxlen = 2000
modelfile = 'speechmap'

(x_train, y_train), (x_test, y_test) = ikrdata.load_audio_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(BatchNormalization(input_shape=(15, 46)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
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
