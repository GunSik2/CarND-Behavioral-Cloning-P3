# Apply Nvidia architecture instead of LENET
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

home_path = '../../CarND-Behavioral-Cloning-P3-sim/'

skip_measurement = 0.1 # Skip steering measurements small

lines = []
dirs=["data", "data-gen/curve", "data-gen/recovery", "data-gen/center", "data-gen/center2"]
for dir in dirs:
    with open(home_path + dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            if(len(lines) is 0):
                print(line)
            #measurement = float(line[3])  # steering_center
            #if(abs(measurement) > skip_measurement):
            lines.append(line)

print('lines =', len(lines))

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
        current_path = home_path + source_path
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement)


    # augment flipping images
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

n_classes = 20
plt.figure(figsize=(12, 5))
plt.hist(measurements, bins=n_classes)
plt.title('Number of measurements')
plt.xlabel('steering')
plt.ylabel('Count')
plt.plot()
plt.show()

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
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1, verbose=2)

model.save('model_a08.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


'''
correction = 0.1
Train on 28342 samples, validate on 7086 samples
Epoch 1/2
82s - loss: 0.0110 - val_loss: 0.0242
Epoch 2/2
76s - loss: 0.0095 - val_loss: 0.0238
dict_keys(['val_loss', 'loss'])
==> success!
'''

'''
- data-gen/center2
Train on 37040 samples, validate on 9260 samples
Epoch 1/2
104s - loss: 0.0172 - val_loss: 0.2543
Epoch 2/2
99s - loss: 0.0155 - val_loss: 0.2319
'''

'''
- dirs=["data", "data-gen/curve", "data-gen/recovery", "data-gen/center", "data-gen/center2"]
Train on 50736 samples, validate on 12684 samples
Epoch 1/2
147s - loss: 0.0209 - val_loss: 0.2042
Epoch 2/2
134s - loss: 0.0191 - val_loss: 0.2060
'''


'''
- dirs=["data", "data-gen/curve", "data-gen/recovery", "data-gen/center", "data-gen/center2"]
Train on 50736 samples, validate on 12684 samples
Epoch 1/1
131s - loss: 0.0204 - val_loss: 0.2015

'''