## 2018-10-11 09:17pm

Changes / configuration:
- Cut 60 and 20 pixels from the top and bottom respectively

Result:
- `5568/5568 [==============================] - 81s 14ms/step - loss: 0.0094 - val_loss: 0.0378`
- The car passes the firt left turn and the bridge after it.
- However it failed to turn left at quite curve turn.

## 2018-10-11 09:17pm

Changes / configuration:
- Reduced the correction value to 0.1

Result:
- `5568/5568 [==============================] - 166s 30ms/step - loss: 0.0112 - val_loss: 0.0495`
- A car drives better, but it could not pass the first turn

## 2018-10-11 08:51pm

Changes / configuration:
- Using left and right camera images to train a model to correct its position
- Correction value is 0.2

Result:
- `5568/5568 [==============================] - 228s 41ms/step - loss: 0.0073 - val_loss: 0.0517`
- A car is steering right too much, need to tune correction value

## 2018-10-11 08:32pm

Changes / configuration:
- Added data augmentation by flipping original images

Result:
- `1856/1856 [==============================] - 73s 39ms/step - loss: 0.0185 - val_loss: 0.0552`
- The car is much more stable now

## 2018-10-11 08:28pm

Changes / configuration:
- Implemented LeNet-5
- Decreased the number of epochs to 3 as val_loss does not decrease

Result:
- `928/928 [==============================] - 37s 40ms/step - loss: 0.0799 - val_loss: 0.0520`
- A car is driving much better, but it is steeling left too much

## 2018-10-11 08:22pm

Changes / configuration:
- Normalized data as `x / 255.0 - 0.5`
- Simplest NN of one flatten layer
- 7 epochs
- Data from `track1_ccw`

Result:
- `928/928 [==============================] - 0s 431us/step - loss: 7062.8762 - val_loss: 5268.6844`