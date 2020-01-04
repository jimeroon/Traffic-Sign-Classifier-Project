#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./stop1.png "Traffic Sign 1 - Stop"
[image5]: ./priority1.png "Traffic Sign 2 - Priority"
[image6]: ./priority2.png "Traffic Sign 3 - Priority"
[image7]: ./yield1.png "Traffic Sign 4 - Yield"
[image8]: ./noentry1.png "Traffic Sign 5 - No Entry"
[image9]: ./speed120.png "Traffic Sign 6 - Speed 120kph"
[image10]: ./nopassing1.png "Traffic Sign 7 - No Passing"
[image11]: ./ahead1.png "Traffic Sign 8 - Ahead Only"
[image12]: ./slippery1.png "Hard Traffic Sign 1 - Slippery Warning"
[image13]: ./snow1.png "Hard Traffic Sign 2 - Snow Warning"
[image14]: ./speed30.png "Hard Traffic Sign 3 - Speed 30 tilted"
[image15]: ./speed130.png "Hard Traffic Sign 4 - Speed 130 not in our set"
[image16]: ./working1.png "Hard Traffic Sign 5 - Working Warning lit sign"
[image17]: ./examples/equalize.jpg "Equalize"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jimeroon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


Here is a link to the [failed color model](https://github.com/jimeroon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Color.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data for the training, test and validation image sets

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I tried to run LeNet in color and was never able to get past 92% with that model.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the images had a wide variety of exposures and I wanted to equalize them to provide more consistent input

![alt text][image17]

I decided not to generate additional data because I had gotten my grayscale model above 94%, and I attempted to add data to the color model and it did not help very much.

I will try and add more data to the the data set to see if I can get the model to match at a higher rate. 

Here is an example of an original image and an augmented image that I tried for the color model:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input 400, outputs 120						|
| RELU					|												|
| Fully connected		| Input 120, outputs 84							|
| RELU					|												|
| Fully connected		| Input 84, outputs 43							|
| Softmax				| Outputs probabilities							|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained two models, one was a color model and this grayscale model. Both models used the AdamOptimizer.
For this grayscale model, I found that slightly smaller batch sizes worked best and chose 50 as my batch size.
I left the learning rate at 0.001 and found that after 28 epochs by model model settle between 94% and 95% validation accuracy.
I used similar hyperparameters for the color model, but it never made it above 92%.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.3%
* validation set accuracy of 94.3%
* test set accuracy of 91.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?   A color model of LeNet was tried first out of a desire to use as much information as possible. The thinking that color information would be better than grayscale ulimately was proved wrong.
* What were some problems with the initial architecture?   The color model suffered from lack of accuracy.
* How was the architecture adjusted and why was it adjusted?   Changing the depth of convultion layers was tried as well as adding dropout to try and combat underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?   Adjustments to batch size and learning rate were tried, both up and down by large and small increments. Once values for batch and rate were found that resulted in growing accuracy, more epochs were added for the best candidates and the models were watched for convergence of accuracy with more epochs.
* What are some of the important design choices and why were they chosen?  Proper preprocessing of the data was critical to help the gradient algorithms minimize error. For example, why might a convolution layer work well with this problem? Convolution layers work well because they distill an image down into relevant subcomponents and feed that data up to the classifier. How might a dropout layer help with creating a successful model? The dropout layer helps the network create redundant paths for recognition, increasing robustness.

If a well known architecture was chosen:
* What architecture was chosen? LeNet was chosen with some good initial hyperparameters because the model has been successful with other image classification tasks.
* Why did you believe it would be relevant to the traffic sign application? classifying numbers is similar to classifying traffic signs. Both types of images typically have a simple symbol on some form of backdrop. The main difference is that with traffic signs, the shape of the sign is relevant, whereas the backdrop is not typically relevant to a number.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The validation and test samples are statistically relevant samples, and the validation sample is kept separate from the training sample. The test sample is never used until the model is fully trained, this helps keep the model from "memorizing" the data set.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Out of curiosity, I tried 13 German traffic signs that I found on the web, 8 easier and 5 very difficult:

The first five were correctly recognized by the model:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

The stop sign was chosen because it has a shadow across the front, which might have confused the model.
 
The next two priority signs were chosen because this type of sign does not have large color contrasts on its face. the thought was that these signs might be confused with stop signs, especially the truncated one.

The yield sign has a large reflection on the bottom and a piece of another sign protruding from the edge. I was curious to see if the convolution layers would try and combine the pieces incorrectly.

The No Entry sign is tilted slightly and has a number of other signs behind it. The model seemed to handle rotation and redundant elements well.


I added three more since the model did well with the first five:

![alt text][image9] ![alt text][image10] ![alt text][image11]

The Speed limit 120kph sign was misread by the model as a 20mph sign, indicating that the model has trouble with dropping some numbers.

The model handled the No Passing sign correctly despite some rotation.

The Ahead Only sign was misread as Ahead And Right. My theory is that the model may need more training on signs with vertical elements.


The next set of 5 images was designed to be very difficult and the model failed on all of them:

These images were all chosen because they were obscured or skewed in some way. Some of these are almost impossible to classify, but I wanted to see how the model would handle them.

![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16]

The first of this set is a Slippery road warning sign almost entirely obscured by snow. The model seemed to have tried to identify by shape and came back with a priority sign.

The next image is similar to the first and is a Freeze Warning sign, also covered in snow and partially obscured. The interesting thing is that the model chose the Right of Way at the Next Crossroads sign, which has an almost star shape in the center similar to the snowflake on the Freeze Warning. 

The 30kph Speed Limit sign is skewed at a large angle and I believe the skew caused the model to classify the sign as a End of speed limit (80km/h).

The 130kph Speed Limit sign is impossible to classify correctly because it is not in our output set of 43 signs. I expected the model to choose one of the other speed limit signs, but it chose the Roundabout sign instead. This is an unfair test but interesting to note that the model rejected it as a speed limit sign.

The final Roadwork sign is from a lighted signboard, where the edges are lit amd the background is dark. The model chose Wild Animals Crossing which is somewhat similar and shares many of the same diagonal elements between the deer and the person with shovel. 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Priority road     	| Priority road 								|
| Priority road     	| Priority road 								|
| Yield					| Yield											|
| No Entry				| No Entry										|
|:---------------------:|:---------------------------------------------:| 
| 120 km/h	      		| 20 km/h					 					|
| No Passing			| No Passing									|
| Ahead Only			| Go straight or right							|
|:---------------------:|:---------------------------------------------:| 
| Slippery (obscured)	| Priority Road      							|
| Freeze (obscured) 	| Right-of-way at the Next Intersection			|
| 30 km/h (skewed) 		| End of speed limit (80km/h)	 				|
| 130 km/h (not exist)	| Roundabout mandatory			 				|
| Road Work	(reversed)	| Wild Animals Crossing					 		|
|:---------------------:|:---------------------------------------------:| 


The model was able to correctly guess 5 of the first 5 traffic signs, which gives an accuracy of 100%. 

The model had a 75% success rate with the first 8 signs, and was fairly close on the ones it missed. 

This compares favorably to the accuracy on the test set of over 94%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is not sure that this is a stop sign (probability of 0.29), and the image does contain a stop sign. 

The top five soft max probabilities for each image are listed below along with the other predictions (* indicates a correct prediction):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|  29.35				| * Stop    									|
|  24.33				| Turn left ahead    							|
|  20.30				| No entry    									|
|  13.07				| Road work    									|
|  11.97				| No passing for vehicles over 3.5 metric tons  |
|:---------------------:|:---------------------------------------------:|
|  52.84				| * Priority road   							|
|  13.81				| No vehicles    								|
|  7.26					| Keep right    								|
|  5.27					| Ahead only   	 								|
|  -2.58				| Speed limit (80km/h)    						|
|:---------------------:|:---------------------------------------------:|
|  73.75				| * Priority road    							|
|  40.17				| Roundabout mandatory    						|
|  12.59				| Right-of-way at the next intersection    		|
|  5.63					| Children crossing    							|
|  2.74					| End of no passing by vehicles over 3.5 m tons |
|:---------------------:|:---------------------------------------------:|
|  151.13				| * Yield    									|
|  15.81				| Speed limit (60km/h)    						|
|  11.78				| Keep right    								|
|  4.80					| Speed limit (80km/h)    						|
|  2.21					| Go straight or right    						|
|:---------------------:|:---------------------------------------------:|
|  81.34				| * No entry    								|
|  43.21				| Speed limit (120km/h)    						|
|  16.28				| Speed limit (60km/h)    						|
|  15.28				| No passing    								|
|  15.24				| No vehicles    								|
|:---------------------:|:---------------------------------------------:|
|  63.21				| Speed limit (20km/h)    						|
|  40.50				| Speed limit (80km/h)    						|
|  36.93				| * Speed limit (120km/h)    					|
|  32.44				| Vehicles over 3.5 metric tons prohibited    	|
|  25.31				| Speed limit (60km/h)    						|
|:---------------------:|:---------------------------------------------:|
|  105.87				| * No passing    								|
|  53.04				| Vehicles over 3.5 metric tons prohibited    	|
|  44.75				| No entry    									|
|  43.45				| No passing for vehicles over 3.5 metric tons  |
|  42.95				| End of no passing    							|
|:---------------------:|:---------------------------------------------:|
|  88.68				| Go straight or right    						|
|  48.94				| * Ahead only    								|
|  48.79				| Speed limit (60km/h)    						|
|  17.34				| Turn right ahead    							|
|  13.88				| Priority road    								|


Hard Image Top 5

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|  26.32				| Priority road    								|
|  15.00				| Bicycles crossing    							|
|  4.76					| Speed limit (80km/h)    						|
|  1.35					| Roundabout mandatory    						|
|  0.40					| No passing for vehicles over 3.5 metric tons  |
|:---------------------:|:---------------------------------------------:|
|  44.77				| Right-of-way at the next intersection    		|
|  40.18				| * Beware of ice/snow    						|
|  18.66				| Double curve    								|
|  13.91				| Road work    									|
|  5.42					| General caution    							|
|:---------------------:|:---------------------------------------------:|
|  51.59				| End of speed limit (80km/h)    				|
|  38.51				| Speed limit (80km/h)    						|
|  31.61				| Roundabout mandatory    						|
|  29.15				| * Speed limit (30km/h)    					|
|  27.87				| Speed limit (100km/h)    						|
|:---------------------:|:---------------------------------------------:|
|  23.97				| Roundabout mandatory    						|
|  20.95				| Speed limit (100km/h)   						|
|  13.85				| Speed limit (80km/h)    						|
|  12.67				| Speed limit (60km/h)    						|
|  11.98				| Speed limit (30km/h)    						|
|:---------------------:|:---------------------------------------------:|
|  32.53				| Wild animals crossing    						|
|  11.04				| * Road work    								|
|  9.67					| Speed limit (60km/h)    						|
|  6.93					| Beware of ice/snow    						|
|  6.10					| Speed limit (80km/h)    						| 


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I was unable to get the Visualizer to work due to syntax error.
