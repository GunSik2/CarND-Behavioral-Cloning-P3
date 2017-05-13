# Apply Nvidia architecture instead of LENET
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # TBD

# center labels in histogram plot
def bins_labels(bins, **kwargs):
    bins[abs(bins) < 1.0e-10] = 0.0  # suppress small value as 0
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    ticks = np.arange(min(bins)+bin_w/2, max(bins), bin_w)
    plt.xticks(ticks, bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

'''
Loading training data: Load training data
'''
home_path = '../CarND-Behavioral-Cloning-P3-sim/'
#skip_measurement = 0.1 # Skip steering measurements small
lines = []
dirs=["data-snip/center", "data-snip/reverse-center", "data-gen/recovery", "data-gen/curve"]
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

print('# of lines =', len(lines))

'''
Loading training data: Create training data using left & right & center images
- left & right measurement values are adjusted for the side camera images
'''
images = []
measurements = []

for line in lines:
    # add three camera images
    for i in range(0, 3):
        correction = 1 # adjusted steering measurements for the side camera images
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

measurements = np.array(measurements)
print(abs(measurements) < 1.0e-8)
print("# of elements < 1.0e-8", np.sum(measurements[abs(measurements) < 1.0e-8]))
print("# of elements < 0.01 ", np.sum(measurements[abs(measurements) < 0.01]))


# histogram
bin_step = 0.05
bins = np.arange(min(measurements), max(measurements) + bin_step, bin_step)
#bins = np.arange(-1.1, 1.1, bin_step)
#plt.figure(figsize=(12, 5))
print(bins)
#bins_labels(bins)
plt.hist(measurements, bins=bins)
plt.title('Number of measurements')
plt.xlabel('steering')
plt.ylabel('Count')
plt.grid(True)
plt.show()
