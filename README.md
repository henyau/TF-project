# **Traffic Sign Recognition**

## Henry Yau
### Goal of the project is to classify German street signs with deeplearning using TensorFlow
---

[//]: # (Image References)

[image1]: ./writeup_images/data_distrubution.png "Visualization"
[image2]: ./writeup_images/data_augmentation.png "Data Augmentation"
[image3]: ./writeup_images/data_augmentation2.png "Data Augmentation"
[image4]: ./writeup_images/130_30.png "Classification on 130km/h"
[image5]: ./writeup_images/data_labels.png "Data Labels"
[image6]: ./writeup_images/Model_accuracy_2.png "Model Accuracy"
[image7]: ./writeup_images/new_signs.png "New Signs"
[image8]: ./writeup_images/softmax_top5_predictions.png "Softmax Top 5"
[image9]: ./writeup_images/new_signs_predictions.png "New Predictions"
[image10]: ./writeup_images/130_30.png "130 km/h"
[image11]: ./writeup_images/LayerInfo.png "Feature map"


### Data Set Summary & Exploration

#### 1. Data summary

The dataset used in this project comes from The German Traffic Sign Recognition Benchmark available at: 
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
The data has been prepackaged as *pickle* files with each image already resized to 32x32 with RGB color channels.

A brief summary of the data:
* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3) (RGB)
* Number of classes = 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set showing an example of each of the classes.
![alt text][image5]

The number of examples for each of the labels for each dataset (test, training, and validation) are shown below to give an idea of the distribution of each class. We see that a few classes are under represented. This knowledge is used later when augmenting the training dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data preprocessing and data augmentation

Though research has shown that there is a difference in using color versus greyscale images in classification algorithms [https://doi.org/10.1371/journal.pone.0029740], here the target are street signs which are designed to be bold and easily recognizable even without color cues. With this assumption that the color information is less significant, we convert the images to greyscale so that with the same number of weights, we may concentrate with a greater emphasis on the shape information.

A final step in the preprocessing is to normalize the data to fit between -1 and 1. This is done so that all features will have equal contribution to the learning and so the gradient descent algorithm will converge more quickly.

Some data augmentation methods are also applied. Data augmentation is used to enlarge the training set by adding modified versions of elements within the training set. This increases the robustness of the trained model to a wider variety of inputs. First, a histogram equalization is applied so that the features of the sign are enhanced. Second, the images are rotated by a randomly determined angle between -25 and 25 degrees. The training, validation and test images seem to be already oriented so using the random rotations may be a hindrance to the learning accuracy. However, when presented with new images sourced from the internet which haven't been adjusted, the performance is expected to be slightly better.

There are many class examples which are under represented. If the number of examples for a label is below a threshold (1000), then an additional example is created. This increases the training set by 10110 examples. One may also add multiple examples derived from the same base image through rotation, translation, scaling or some other image manipulation.

The image preprocessing steps and data augmentation can be seen below, from left to right, the original image, converted to greyscale, histogram equalization, and finally a random rotation.
![alt text][image3]






#### 2. Learning model

The learning model is based on the well known LeNet-5 model proposed by LeCun et al. for use in classifying hand writen digits. A dropout step is used in the final two fully connected layers.  

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| ReLu					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24
|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs Output 10x10x48|
| ReLu					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x48 
|						|								
| Flatten               | Output 1200x1
| Fully Connected      	| Output 256
| ReLu					|												|
| Dropout				| Keep probability of 0.7	
|
| Fully Connected      	| Outputs 96x1 	|
| ReLu					|												|
| Dropout				| Keep probability of 0.7					    |
| Fully Connected      	| Outputs 43 (number of classes) 	|
| Softmax		      	| 


#### 3. Model Training

The model is trained using a minibatch size of 48 run for 25 epochs with the Adams optimizer with a learning rate of 0.001. For the cost functional, both a cross entropy and a hinge loss function were tested (comparing the output logits with the one hot labels). Their performance was roughly the same.



#### 4. Improving the validation set accuracy to a minimum of 0.93

The model used was derived from LeNet-5 which was originally used to train hand writen digits. Traffic sign classification is a similar problem so one might assume that the general architecture of the model can be applied. However, traffic signs have significantly more features than hand writen digits. So the first step in improving the accuracy of the model was increasing the number of filters used in the convolution layers. The next step in improving the accuracy was to include dropout in the final two fully connected layers. Dropout is a simple method to prevent the model from overfitting the training data. With these two addition alone the validation set accuracy was already pushed to above 0.95.

The next step was to improve the quality of the training set data. As stated above, there are many classes which have too few samples to adequately learn. This can be somewhat rectified by augmenting the training data with samples constructed from the same set. Using various image manipulation methods which still perserve the core information of the sign (such as limited rotation, translation or histogram manipulation), a new example can be created. Using this augmented dataset, the validation accuracy was pushed further to above 0.97.

The learning progression can be seen below:

![alt text][image6]

The final model results after 25 epochs were:
* training set accuracy of 99.91%
* validation set accuracy of 97.2% 
* test set accuracy of 95.7%

As the training set accuracy is nearly 100 percent, we can infer that the model has learned that set completely. If the training set was a good representation of what we might expect in the validation and test sets, then the evaluation of those sets should have high accuracy as well. Due to the stochastic nature of the model (from stochastic gradient descent in the optimizer to the random dropouts), each run is slightly different. However, if the model's accuracy is relatively consistant then we can infer that the model learning appropriately.

### Testing the Model on New Images

#### 1. Image of traffic signs sourced from the internet

Fifteen German traffic sign images from the internet were tested on the model. These images are shown below after they have been converted to grayscale and resized to 32x32:

![alt text][image7]

The image interpolator in OpenCV appears to do a poor job at minification. Both the cubic and linear interpolators have severe aliasing artifacts. Each image may provide some other difficulties for the classifier model. These difficulties are listed below:

 1. The first image is of a 130km/h sign which is not a class in the training set. There is a 30km/h sign and a 100km/h sign. I was curious to see what the classifier would produce. 
 2. Sign has wording below which may confuse the classifier
 3. Image has a watermark
 4. Image has watermark and is shot from off center
 5. Image slightly askew
 6. Low quality image with JPG artifacts
 7. Building in background
 8. Sign is covered in markings
 9. Lighting is not uniform. Much darker on top
 10. Busy background
 11. Low quality image
 12. Image does not have square aspect ratio, rather than cropping the images, they are resized which introduce distortions
 13. Image has watermarks
 14. Busy background
 15. Watermark and askew


#### 2. Predictions on new test images

Here are the results of the prediction:

![alt text][image9]

There is one misclassified image, the 130km/h sign which was not in part of the training set. Therefore all the images were classified correctly. The last image, the wildlife crossing sign, was not classified correctly until the training set was augmented with the rotated additions.

#### 3. Softmax probabilities
The top five softmax probabilities are shown for each image as a bar graph below:
![alt text][image8]


The 130km/h sign has been labeled with almost complete certainty that it is a 30km/h sign. However, without using the histogram equalization preprocessing and using only the original data set, the model had predicted a chance the image represented a 100km/r sign as shown below:
![alt text][image10]


### (Optional) Visualizing the Neural Network
#### 1.Feature map visualization

Below are the feature maps of the first convolutional layer predicting the second image (general caution):

![alt text][image11]

We can attempt to identify what the trained model deems as important. Most prominantly is the silouette of the sign itself. This is a feature which is seen in all the feature maps. Next the features within the signs are somewhat segregated within each map. The exclaimation point can be seen quite clearly in various maps.
