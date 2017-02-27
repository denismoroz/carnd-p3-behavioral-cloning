#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_driving]: ./images/center.jpg 
[recover_start]: ./images/recover_start.jpg 
[recover_return]: ./images/recover_return.jpg
[recover_end]: ./images/recover_end.jpg
[flip]: ./images/flip.jpg
[flip_result]: ./images/flip_result.jpg
[flip_brightness]: ./images/flip_brightness.jpg
[model]:./images/model.png

[t1_pre_processing]:./images/t1_preprocess_data.png
[t1_post_processing]:./images/t1_post_processing.png

## Video of model driving Udacity Simulator on both tracks

* [Track 1](https://www.youtube.com/watch?v=aFap9xOKnDU)
* [Track 2 version 1](https://www.youtube.com/watch?v=KeqglS0U8gM)
* [Track 2 version 2](https://www.youtube.com/watch?v=aX07OHAEb-8)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* t1_auto.mp4 example of running this model on the first track in simulator

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 40-44) 

The model includes ELU layers to introduce nonlinearity (code line 38), and the data is normalized in the model using a Keras lambda layer (code line 35). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 48, 51, 54, 57). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 197). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 61). As a learning rate was taken 0.0001 after several iterations and tests.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Important thing was to provide model images set where car is returned back to the center of the line on the road if steering wheel angle was predicted wrong previously.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use five convolutional layers and three fully connected layers.

My first step was to use a convolution neural network model similar to the [NVIDIA convolutional network for predicting steering angle](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because it is already tested and showed good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added several dropout layers to model and collect more data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like difficult turning rights or places where road border ends. To improve the driving behavior in these cases, I recorded extra data, trying to drive on these places as smooth as possible.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 28-61) consisted of a convolution neural network with the following layers and layer sizes:
  * Input (None, 160, 320, 3)
  * cropping2d    -> (None, 100, 250, 3)
  * lambda        -> (None, 100, 250, 3)
  * convolution2d -> (None, 48, 123, 24) 
  * convolution2d -> (None, 22, 60, 36) 
  * convolution2d -> (None, 9, 28, 48) 
  * convolution2d -> (None, 7, 26, 64) 
  * convolution2d -> (None, 5, 24, 64) 
  * flatten       -> (None, 7680)
  * dropout       -> (None, 7680) 
  * dense         -> (None, 700)  
  * dropout       -> (None, 700) 
  * dense         -> (None, 70) 
  * dense         -> (None, 1)  
  
Here is a visualization of the architecture:

![alt text][model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving using mouse and a single track using keyboard only to control car. Here is an example image of center lane driving:

![alt text][center_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to return back to road after wrong prediction:

![alt text][recover_start]
![alt text][recover_return]
![alt text][recover_end]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help model train one more track. For example, here is an image that has then been flipped:

![alt text][flip]
![alt text][flip_result]


On a final step of reprocessing I was changing brightness randomly for image.

![alt text][flip]
![alt text][flip_brightness] 


After the collection process, I had 9 number of data points for both tracks. I then preprocessed this data by joining all data sets.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Resulting data set I used for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by several iterations and notice that validation loss and training lose does not decreased greatly even number of epochs increased. I used an adam optimizer so that manually training the learning rate wasn't necessary.
