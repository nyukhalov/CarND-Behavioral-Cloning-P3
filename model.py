from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, MaxPooling2D
from scipy import ndimage
import numpy as np
import csv
import cv2

# Importing data
data_root_path = './data/'
folders = [
    'track1_ccw'
]

lines = []
images = []
measurements = []

for folder in folders:
    data_path = data_root_path + folder
    print('Importing data from %s' % data_path)
    with open(data_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = data_path + '/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

# Defining a model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(filters=6, kernel_size=5))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=5))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Compilation
model.compile(optimizer='adam', loss='mse')

# Training
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
