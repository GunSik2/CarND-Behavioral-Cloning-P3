# Use all three camera images to train the model, so
# feed the left and right camera images to your model as if they were coming from the center camera.
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

    #augment flipping images
    #image_flipped = np.fliplr(image)
    #measurement_flipped = -measurement
    #images.append(image_flipped)
    #measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

# normarlized image
model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=2)

model.save('model_cameras.h5')

'''
correction = 0.3, epochs=3
Train on 19286 samples, validate on 4822 samples
Epoch 1/3
69s - loss: 0.3640 - val_loss: 0.0127
Epoch 2/3
63s - loss: 0.0100 - val_loss: 0.0112
Epoch 3/3
63s - loss: 0.0084 - val_loss: 0.0113
'''

'''
correction = 0.1
Train on 19286 samples, validate on 4822 samples
Epoch 1/5
65s - loss: 0.1743 - val_loss: 0.0123
Epoch 2/5
61s - loss: 0.0097 - val_loss: 0.0117
Epoch 3/5
61s - loss: 0.0081 - val_loss: 0.0116
Epoch 4/5
60s - loss: 0.0070 - val_loss: 0.0123
Epoch 5/5
60s - loss: 0.0063 - val_loss: 0.0126
'''

'''
correction = 0.05
Train on 19286 samples, validate on 4822 samples
Epoch 1/5
66s - loss: 0.3939 - val_loss: 0.0118
Epoch 2/3
61s - loss: 0.0100 - val_loss: 0.0107
Epoch 3/3
64s - loss: 0.0089 - val_loss: 0.0104
'''

'''
correction = 0.4
Train on 19286 samples, validate on 4822 samples
Epoch 1/5
64s - loss: 0.2442 - val_loss: 0.0130
Epoch 2/5
61s - loss: 0.0098 - val_loss: 0.0126
Epoch 3/5
62s - loss: 0.0082 - val_loss: 0.0123
Epoch 4/5
61s - loss: 0.0071 - val_loss: 0.0126
Epoch 5/5
'''