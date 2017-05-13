# Cropping images, so
# your model might train faster if you crop each image to focus on only the portion of the image that is useful for predicting a steering angle.

import csv
import cv2
import numpy as np

home_path = '../../CarND-Behavioral-Cloning-P3-sim/'

lines = []
dirs=["data"] #, "data-gen/curve", "data-gen/recovery"]
for dir in dirs:
    with open(home_path + dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            if(len(lines) is 0):
                print(line)
            lines.append(line)

print('lines =', len(lines))

images = []
measurements = []
for line in lines:
    # add three camera images at center, left, right
    for i in range(3):
        correction = 0.1 # adjusted steering measurements for the side camera images
        measurement = float(line[3]) # steering_center
        if i is 1: # left
            measurement += correction
        elif i is 2: # right
            measurement -= correction

        source_path = line[i] # 0 - center, 1 - left, 2 - right
        current_path = home_path + source_path

        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement)

        #augment flipping images
        #image_flipped = np.fliplr(image)
        #measurement_flipped = -measurement
        #images.append(image_flipped)
        #measurements.append(measurement_flipped)

print('#images = ', len(images), ' #measurements = ', len(measurements))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# normarlized image
model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))  # `((top_crop, bottom_crop), (left_crop, right_crop))`
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1, verbose=1)

model.save('model_cropping_more.h5')

'''
Train on 19286 samples, validate on 4822 samples
Epoch 1/3
59s - loss: 0.0327 - val_loss: 0.0119
Epoch 2/3
51s - loss: 0.0103 - val_loss: 0.0129
Epoch 3/3
50s - loss: 0.0098 - val_loss: 0.0126
'''
'''
(+ Flipping images)
Train on 38572 samples, validate on 9644 samples
Epoch 1/1
106s - loss: 0.0269 - val_loss: 0.0126
'''