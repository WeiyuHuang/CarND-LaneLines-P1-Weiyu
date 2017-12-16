# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 

First, I converted the images to grayscale. 

Second, I passed the grayscale image through a Gaussian kernel with kernel size 5 to smooth the image. 

Third, I used Canny edge detection algorithm with low threshold being 100 and high threshold being 200 to detect edges in the smoothed image. The thresholds were chosen slightly higher than what were used in the previous quiz to avoid picking edges of other cars.

Fourth, a region selection mask was applied to make our focus on the bottom part of the image. I used a triangle formed by the lower left corner, the center of the image, and the lower right corner, because this area well represents the expected area of the lanes.

Fifth, I applied Hough transform with \rho being 2, \theta being 1 in degree, threshold being 30, min line len being 15, and max line gap being 5. The values were chosen very similar to what were used in the previous quiz except for max line gap, for the reason that we want to keep the small edge corresponding to dotted lanes.

Sixth, I combined the detected lines with the original image. 

This pipeline result in good performance. The next step is to combine lines into thick long lines corresponding to the left and right lanes.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to perform the following three tasks.
1. For all lines, cluster them into two groups, one corresponding to positive slopes, and another corresponding to negative slopes. Lines whose absolute value of slopes are larger than 4 or are smaller than 0.15 are ignored, because these values are not the expected slopes of lanes.
2. For lines with positive slope, compute a weighted average of their slopes and intercepts, where weight corresponds to the length for each of the line. Perform similarly for lines with negative slope.
3. Compute x values for the positive line with y values corresponding to the bottom of the image and center of the image. Compute x and y values for the negative line with y values at these two endpoints as well. Draw lines using these x and y values

The final output is like
![alt text][image1]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is that currently we do not apply smoothing in the time domain. Intuitively, the lanes identified in the previous frame should be very similar to the lanes identified in the next frame. 

Another shortcoming is that the region selection is currently hard coded. If unexpected situations happened, e.g. sharp turns, rotation of cameras, the entire algorithm will break.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to apply a smoothing in the time domain to increase the robustness of lines detected.

Another potential improvement could be to apply a dynamic choosing algorithm in region selection. A potential pipeline works as follows:
1. Remove pixels correspond to the sky
2. Run lane detection
3. Remove small lines
4. Focus on the region with active lines
5. Rerun lane detection
