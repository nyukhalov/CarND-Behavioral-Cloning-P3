from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
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

X_train = np.array(images)
y_train = np.array(measurements)

# Defining a model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

# Compilation
model.compile(optimizer='adam', loss='mse')

# Training
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')
