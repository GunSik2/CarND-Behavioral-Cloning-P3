# Apply Nvidia architecture instead of LENET
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


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
    # add three camera images
    for i in range(3):
        correction = 0.1 # adjusted steering measurements for the side camera images
        measurement = float(line[3]) # steering_center
        if i is 1: # left
            measurement += correction
        elif i is 2: # right
            measurement -= correction

        source_path = line[i] # 0 - center, 1 - left, 2 - right
        filename = source_path.split('/')[-1]
        current_path = home_path + 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement)

    # augment flipping images
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)


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

model.add(Conv2D(24, (5, 5), subsample=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), subsample=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), subsample=(2,2), activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=2)

model.save('model_nvidia.h5')

'''
correction = 0.1
Train on 19286 samples, validate on 4822 samples
Epoch 1/3
56s - loss: 0.0118 - val_loss: 0.0115
Epoch 2/3
50s - loss: 0.0101 - val_loss: 0.0116
Epoch 3/3
50s - loss: 0.0097 - val_loss: 0.0114
'''
