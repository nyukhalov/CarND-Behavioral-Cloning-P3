# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [writeup_report.md](writeup_report.md) summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I did not implement data generator as the virtual machine I used for training had enough memory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the model proposed by NVIDIA developers in the [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) article.

It consists of five convolutional layers followed by three fully-connected layers. Each layer is described in the table below:

|Layer | Output Shape | Configuration |
| ---  | ------------ | ------------- |
|Input           | 65x320x3  | Normalized input |
|Convolutional   | 31x158x24 | filters=24, kernel=5x5, strides=2x2 |
|Activation      | 31x158x24 | RELU |
|Convolutional   | 14x77x36  | filters=36, kernel=5x5, strides=2x2 |
|Activation      | 14x77x36  | RELU |
|Convolutional   | 5x37x48   | filters=48, kernel=5x5, strides=2x2 |
|Activation      | 5x37x48   | RELU |
|Convolutional   | 3x35x64   | filters=64, kernel=3x3, strides=1x1 |
|Activation      | 3x35x64   | RELU |
|Convolutional   | 1x33x64   | filters=64, kernel=3x3, strides=1x1 |
|Activation      | 1x33x64   | RELU |
|Flatten         | 2112      | |
|Fully connected | 100       | |
|Activation      | 100       | RELU |
|Fully connected | 50        | |
|Activation      | 50        | RELU |
|Fully connected | 10        | |
|Activation      | 10        | RELU |
|Output          | 1         | |

The network has 348219 parameters.

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting I've collected data from both track 1 and 2.
In addition, I applied data augmentation by adding a horizontally flipped version of each frame. Having so various data set was enough to successfully train the model. 

I split the collected data into trained and validated sets (as 80% and 20% respectively). The model was trained and validated using the data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track 1 as well as on the track 2.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving without any recovering driving. However, I used images from the left and right cameras to train the model to recover. For more details see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Simulation

todo