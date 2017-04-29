import numpy as np
np.random.seed(1337)  # not sure if this helps

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import ikrdata

K.set_image_data_format('channels_first')
batch_size = 64
num_classes = 31
epochs = 32
img_rows, img_cols = 80, 80
input_channels = 3
modelfile = 'tenecaf'

(x_train, y_train), (x_test, y_test) = ikrdata.load_graphic_data()

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# training data augmentation
datagen = ImageDataGenerator(
    rotation_range=3,
    #zoom_range=0.05,
)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(input_channels, img_cols, img_rows)))
model.add(Conv2D(
    28,
    kernel_size=(4, 4),
    activation='relu',
    bias_initializer=initializers.constant(0.1),
))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(
    512,
    activation='relu',
    bias_initializer=initializers.constant(0.1),
))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size,
                 #save_to_dir='augs/'
                 ),
    steps_per_epoch=len(x_train)/batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if input('Save model? (y): ').lower() == 'y':
    path = '{}-acc{}.h5'.format(modelfile, score[1])
    model.save(path)
    print('Model saved to', path)
