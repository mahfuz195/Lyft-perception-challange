# Lyft Perception Challange 2018 (My Approach)

From May 1, 2018 to June 3, 2018 Lyft arranged a percention challange for Udacity self driving nano-degree current and formar students. Participents were encoursed to apply computer vision and deep learning based approach to solve the given problem. As a student of Term 2 of Self driving nano-degree program of Udacity, I participated in the challange and placed 37th position globally.  As a newbi in deep learning, I had zero knowledge about image segmentaiton. But after a month long hard work, I was able to pull off a decent result. In this document, I am going to describe my approach to the problem.

## Leader Board

![alt text](leader_board.png)

## Challenge Overview
The goal in this challenge is pixel-wise identification of objects in camera images. In other words, the task is to identify exactly what is in each pixel of an image! More specifically, you need to identify cars and the drivable area of the road. The images below are a simulated camera image on the left and a label image on the right, where each different type of object in the image corresponds to a different color.
<p>
    <img src="challange.png" alt>
    <em>Example of a simulated camera image (left) and pixel-wise labels of cars, road, lane markings, pedestrians, etc. (right) </em>
</p>

## Dataset

The given dataset by lyft contained 1000 images containing road and cars along with ground truth (segmented image) from CARLA autonomous vehicle simulator. 

## Model Selection

First I implemented the SegNet Architecture. But my model as not performing well and was giving a very low framerate and accuracy. I tried data augmentation, dropout, and regularization on SegNet. But still model was performing poorly. Then I realized that the model was too complicated and selected Fully Convolution Network (FCN) [Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) as it is simpler model. I modified VGG16 pertained model to segment the cars and road using FCN-8 for this competition. I selected FCN-8 as it provides better accuracy compared to FCN-16 and FCN-32. I modified the output layers for 3 classes so that it predicts Cars, Roads, and None for rest of the classes. My model takes input (352, 800, 3) as the input image and outputs an image with the same size. 


## Data Prepossing

... coming soon ...

## Data Augmentation

... coming soon ...

## Framerate Improvement

... coming soon ...

## Lesson Learnt and Further Improvement

... coming soon ...

## Acknowledgements

... coming soon ...

