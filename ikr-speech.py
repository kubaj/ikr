from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
import ikrdata

batch_size = 32
num_classes = 31
epochs = 64
maxlen = 2000
modelfile = 'speechmap'

(x_train, y_train), (x_test, y_test) = ikrdata.load_audio_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(x_train.shape[0], 26, 26, 1)
x_test = x_test.reshape(x_test.shape[0], 26, 26, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(
    24, (3, 6),
    activation='tanh',
    bias_initializer=initializers.constant(0.1),
    input_shape=(26, 26, 1),
))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid', bias_initializer=initializers.constant(0.1)))
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
