from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D
from scipy import ndimage
import numpy as np
import csv
import cv2

"""
# Download data
wget -O data.zip https://www.dropbox.com/s/u7zy7414mu6vycs/carnd-behavioral-cloning-data2.zip?dl=0

# Unzip the archive
unzip data.zip -d ./data
"""

# Importing data.
# Each folder contains recorded images of a single run.
# 'track1_*' folders contain recorded images from runs in the track one.
# 'track2_*' folders contain recorded images from runs in the track two.
data_root_path = './data/'
folders = [
    'track1_ccw',
    'track1_ccw_2',
    'track1_ccw_turnA',
    'track1_ccw_turnB',
    'track1_cw',
    'track1_cw_2',
    'track1_cw_turnA',
    'track1_cw_turnB',
    'track2_cw_mouse',
    'track2_cw_mouse_2',
    'track2_ccw_mouse',
    'track2_ccw_mouse_2'
]

## Steering angle is in between -1 and 1 inclusive
## -1 implies the most left position (a car turns left)
##  1 implies the most right position (a car turns right)
steering_correction = 0.1

images = []
measurements = []

def load_image(data_path, source_image_path):
    """
    Load an image from filesystem as an numpy array.
    `data_path` - path to a folder containg images
    `source_image_path` - original path of the loading image
    """
    filename = source_image_path.split('/')[-1]
    current_path = data_path + '/IMG/' + filename
    return ndimage.imread(current_path)    

def flip_image(image, steering):
    """
    Flips the `image` horizontally.
    """
    image_flipped = np.fliplr(image)
    steering_flipped = -steering
    return (image_flipped, steering_flipped)

# Loading images from all specified folders
for folder in folders:
    lines = []
    data_path = data_root_path + folder
    print('Importing data from %s' % data_path)
    with open(data_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # For each line in the csv-file
    # 1. Load center, left and right images
    # 2. Calculate corrected steering angle for the left and right images
    # 3. Add flipped images
    for line in lines:
        center_image_path = line[0]
        left_image_path = line[1]
        right_image_path = line[2]

        image_center = load_image(data_path, center_image_path)
        image_left = load_image(data_path, left_image_path)
        image_right = load_image(data_path, right_image_path)
        
        steering_center = float(line[3])
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        
        images.extend([image_center, image_left, image_right])
        measurements.extend([steering_center, steering_left, steering_right])

        image_center_flipped, steering_center_flipped = flip_image(image_center, steering_center)
        image_left_flipped, steering_left_flipped = flip_image(image_left, steering_left)
        image_right_flipped, steering_right_flipped = flip_image(image_right, steering_right)

        images.extend([image_center_flipped, image_left_flipped, image_right_flipped])
        measurements.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])

X_train = np.array(images)
y_train = np.array(measurements)

# Defining a model
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2,2), activation="relu"))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2,2), activation="relu"))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2,2), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

model.summary()

# Compilation
model.compile(optimizer='adam', loss='mse')

# Training and saving the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')
