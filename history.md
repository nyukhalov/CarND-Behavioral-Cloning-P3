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