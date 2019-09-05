# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.jpg "Visualization"
[image2]: ./writeup_images/grayscale.jpg "Grayscaling"
[image3]: ./writeup_images/jitter.jpg "Jitter"
[image4]: ../data/web/11.jpg "Right of Way"
[image5]: ../data/web/12.jpg "Priority"
[image6]: ../data/web/13.jpg "Yield"
[image7]: ../data/web/18.jpg "Warning"
[image8]: ../data/web/25.jpg "Road Work"
[image9]: ./writeup_images/prediction.jpg "Jitter"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Numpy to calculate the summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a comparison of histograms showing the distribution of the various classes in the dataset. Since the distributions are similar, the model trained with the training set should have good accuracy with respect to the validation and test sets. Some classes have more examples than that of the other classes. The model will be more accurate for the classes with more examples.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because, according to [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the CNN performed better without color. This was true for my case as well.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

For the second step, I normalized the image data because having data at zero mean and equal variance would greatly assist the optimizer. I normalized the grayscale values from -1 to 1 to align with the `(pixel - 128)/ 128` formula.

Lastly, I decided to generate additional data because, according to [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) again, it helps to have additional data that would account for the variation that may appear in the test set.

To add more data to the the data set, I used the following techniques ("jittering") from [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) with OpenCV:
1. Translate the image randomly within -2 to 2 pixels in both the x and y direction
2. Rotate and scale the image randomly within -15 to 15 degrees and 0.9 to 1.1 ratios 

Here is an example of an original image and an augmented image:

![alt text][image3]

The size of the augmented data set was 5 times the size the original data set. The agumented data increased the differences in number of images of each class.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model (based on LeNet-5) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| 0.5 keep prob 								|
| Max pooling	      	| 2x2 ksize, 2x2 stride, outputs 14x14x6 		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				| 0.5 keep prob 								|
| Max pooling	      	| 2x2 ksize, 2x2 stride, outputs 5x5x16 	 	|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 43									|
| Softmax				|         										|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, 128 batch size, 10 epochs, 0.001 learning rate, and 0.5 keep probability. (All are defaults for LeNet lab and other quizes).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the results is located in the 13th cell of the jupyter notebook.

My final model results were:
* training set accuracy of 0.979
* validation set accuracy of 0.934
* test set accuracy of 0.919

I used an iterative approach:
* LeNet-5 was the first architecture that was tried because the template notebook stated that it was a good starting point for the project.
* Some problems with the initial architecture include mediocre validation accuracy with the default training set and overfitting
* The architecture was adjusted with dropouts after each activation. This was done to reduce the overfitting.
* Interestingly, after preproscessing the data, no parameters needed to be tuned to achieve 93% validation accuracy.
* Convolution layer worked well with this problem because it allowed for extraction of low level features (lines and curves) and building up of high level features (shapes). Dropout layer helped with preventing overfitting to the training set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and edited to be 32x32:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

1. The first image might be difficult to classify because there is a portion of another sign below it. 
2. The second image might be difficult to classify because there is another round sign behind it and an additional object in the background. 
3. The third image might be difficult to classify because of the green background. 
4. The fourth image might be difficult to classify because there is text below the sign.
5. The fifth image might be difficult to classify because of the shadow on the sign and a tree in the background. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of Way			| Right of Way									|
| Priority Road			| Speed Limit (50km/h)  						| 
| Yield     			| Yield 										|
| General Caution     	| General Caution								|
| Road Work	      		| Wild Animal Crossing			 				|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is significantly lower than the test set accuracy of 91.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 9th and 11th cell of the jupyter notebook. The top five soft max probabilities were given below:

![alt text][image9]

1. For the first image, the model is relatively sure that this is a "General Caution" sign (probability of 0.707), and the image does contain a "General Caution" sign. 

2. For the second image, the model is absolutely sure that this is a "Yield" sign (probability of 1.0), and the image does contain a "Yield" sign.

3. For the third image, the model is absolutely sure that this is a "Right of Way" sign (probability of 1.0), and the image does contain a "Right of Way" sign.

4. For the fourth image, the model is very sure that this is a "Wild Animal Crossing" sign (probability of 0.94), but the image contains a "Road Work" sign (which is the second best prediction with probability of 0.06).

5. For the fifth image, the model is very sure that this is a "Speed Limit (50km/h)" sign (probability of 0.992), but the image contains a "Priority" sign (which is the second best prediction with probability of 0.004).

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


