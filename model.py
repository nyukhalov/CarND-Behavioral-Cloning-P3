from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from scipy import ndimage
import numpy as np
import csv
import cv2

# Importing data
lines = []
with open('./data/track1_ccw/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/track1_ccw/IMG/' + filename
    #image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# Defining a model
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

# Compilation
model.compile(optimizer='adam', loss='mse')

# Training
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')
