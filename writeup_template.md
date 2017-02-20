#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[center_driving]: ./images/center.jpg 
[recover_start]: ./images/recover_start.jpg 
[recover_return]: ./images/recover_return.jpg
[recover_end]: ./images/recover_end.jpg
[flip]: ./images/flip.jpg
[flip_result]: ./images/flip_result.jpg
[flip_brightness]: ./images/flip_brightness.jpg
[model]:./images/model_layers.png
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Important thing was to provide model image set where car is returned back to road if steering wheel angle was predicted wrong previously.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use several convolutional layers and fully connected layers.

My first step was to use a convolution neural network model similar to the [NVIDIA convolutional network for predicting steering angle](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because it is already tested and showed good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added several dropout layers to model.

Then after several iterations, I decided to switch to ELU activation layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like difficult turning rights or places where road border ends. To improve the driving behavior in these cases, I recorded extra data, trying to drive on these places as smooth as possible.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 28-61) consisted of a convolution neural network with the following layers and layer sizes:
 
  * cropping2d  160x320x3
  * lambda_2    63x284x3
  * convolution2d 30x140x24
  * convolution2d 13x68x36
  * convolution2d 5x32x48
  * convolution2d 3x30x64
  * convolution2d 1x28x64
  * flatten_2 1792
  * dropout   1792
  * dense   1164 
  * dropout 1164 
  * dense   800 
  * dropout 800
  * dense   400 
  * dropout 400
  * dense   100
  * dense   1 
  
  
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving using mouse and a single track using keyboard only to control car. Here is an example image of center lane driving:

![alt text][center_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to return back to road after wrong prediction. :

![alt text][recover_start]
![alt text][recover_return]
![alt text][recover_end]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help model train one more track. For example, here is an image that has then been flipped:

![alt text][flip]
![alt text][flip_result]


On a final step of reprocessing I was changing brightness of the image.

![alt text][flip]
![alt text][flip_brightness] 


After the collection process, I had 9 number of data points. I then preprocessed this data by joining all these data sets.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by several iteration and notice that validation loss and training lose does not decreased greatly even increase number of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
