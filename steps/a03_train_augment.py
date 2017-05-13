# Data augmentation to help with the left turn bias
# flipping images and taking the opposite sign of the steering measurement. For example:
#       import numpy as np
#       image_flipped = np.fliplr(image)
#       measurement_flipped = -measurement

import csv
import cv2
import numpy as np

home_path = '../../CarND-Behavioral-Cloning-P3-sim/'

lines = []
with open(home_path + 'data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        if(len(lines) is 0):
            print(line)
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = home_path + 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    #augment flipping images
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# normarlized image
model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=2)

model.save('model_aug.h5')

'''
Train on 12857 samples, validate on 3215 samples
Epoch 1/5
47s - loss: 0.1307 - val_loss: 0.0128
Epoch 2/5
42s - loss: 0.0110 - val_loss: 0.0123
Epoch 3/5
43s - loss: 0.0091 - val_loss: 0.0123
Epoch 4/5
43s - loss: 0.0081 - val_loss: 0.0125
Epoch 5/5
43s - loss: 0.0073 - val_loss: 0.0126
'''