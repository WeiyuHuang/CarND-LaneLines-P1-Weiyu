#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

print('Started')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def compute_x_from_y(y, b, m):
    """
    Compute x from y given slope b and intercept m
    """
    return (y - m) / b


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    """
    pos_b = []  # slope for lines with positive slopes (the line on the right)
    pos_m = []  # intercept for with positive slop (the line on the left)
    pos_weight = []  # longer lines have higher weights for contribution
    neg_b = []  # similarly, for lines with negative slopes (on the left)
    neg_m = []
    neg_weight = []

    threshold_small = 0.15  # smallest absolute value of slopes considered
    threshold_large = 4  # largest absolute value of slopes considered

    # separate lines into positive and negative slopes, record slopes, intercept, and weight
    for line in lines:
        l = line[0]
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
        slope = (y2 - y1) / (x2 - x1)
        if (threshold_small < slope) and (slope < threshold_large):  # positve
            pos_b.append(slope)
            pos_m.append(y1 - slope * x1)
            pos_weight.append(np.linalg.norm([x1 - x2, y1 - y2]))
        if (-threshold_small > slope) and (slope > -threshold_large):  # negative
            neg_b.append(slope)
            neg_m.append(y1 - slope * x1)
            neg_weight.append(np.linalg.norm([x1 - x2, y1 - y2]))

    # weighted average for positive and negative
    pos_b_combined = np.sum([a*b for a, b in zip(pos_weight, pos_b)]) / np.sum(pos_weight)
    pos_m_combined = np.sum([a*b for a, b in zip(pos_weight, pos_m)]) / np.sum(pos_weight)
    neg_b_combined = np.sum([a*b for a, b in zip(neg_weight, neg_b)]) / np.sum(neg_weight)
    neg_m_combined = np.sum([a*b for a, b in zip(neg_weight, neg_m)]) / np.sum(neg_weight)

    # compute x, y for bottom of screen and middle of screen
    imshape = img.shape
    pos_y1, neg_y1 = imshape[0], imshape[0]  # lower part
    pos_y2, neg_y2 = 0.6 * imshape[0], 0.6 * imshape[0]  # upper part
    pos_x1 = compute_x_from_y(pos_y1, b=pos_b_combined, m=pos_m_combined)
    pos_x2 = compute_x_from_y(pos_y2, b=pos_b_combined, m=pos_m_combined)
    neg_x1 = compute_x_from_y(neg_y1, b=neg_b_combined, m=neg_m_combined)
    neg_x2 = compute_x_from_y(neg_y2, b=neg_b_combined, m=neg_m_combined)

    # add lines
    cv2.line(img, (int(pos_x1), pos_y1), (int(pos_x2), int(pos_y2)), color, thickness)
    cv2.line(img, (int(neg_x1), neg_y1), (int(neg_x2), int(neg_y2)), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


#############################################
################# I M A G E #################
#############################################
image_files = os.listdir('test_images')

for image_file in image_files:
    # load image, convert to grayscale
    image = mpimg.imread('test_images/' + image_file)
    gray = grayscale(image)

    # gaussian kernal smoothing
    post_gaussian_blur = gaussian_blur(gray, kernel_size=5)

    # Canny edge detection
    edges = canny(post_gaussian_blur, low_threshold=100, high_threshold=200)

    # region selection
    imshape = image.shape
    vertices = np.array([[(0.05*imshape[1], imshape[0]),  # bottomleft
                          (0.5*imshape[1], 0.55*imshape[0]),  # top
                          (0.95*imshape[1], imshape[0])]],  # bottomright
                        dtype=np.int32)  # bottomright
    masked_edges = region_of_interest(edges, vertices)

    # Hough transform and line addition
    line_image = hough_lines(masked_edges, rho=2, theta=1*np.pi/180,
                             threshold=30, min_line_len=15, max_line_gap=5)

    # combine lines with the original image
    combo = cv2.addWeighted(image, 0.9, line_image, 1, 0)
    plt.imshow(combo)

    # save
    mpimg.imsave('test_images_output/' + image_file, combo)

#############################################
################# V I D E O #################
#############################################

# play with lf_image
def invert_green_blue(image):
    return image[:,:,[0,2,1]]


def process_image(image):
    gray = grayscale(image)

    # gaussian kernal smoothing
    post_gaussian_blur = gaussian_blur(gray, kernel_size=5)

    # Canny edge detection
    edges = canny(post_gaussian_blur, low_threshold=100, high_threshold=200)

    # Region selection
    imshape = image.shape
    vertices = np.array([[(0.05 * imshape[1], imshape[0]),  # bottomleft
                          (0.5 * imshape[1], 0.55 * imshape[0]),  # top
                          (0.95 * imshape[1], imshape[0])]],  # bottomright
                        dtype=np.int32)  # bottomright
    masked_edges = region_of_interest(edges, vertices)

    # Get lines
    line_image = hough_lines(masked_edges, rho=2, theta=1 * np.pi / 180,
                             threshold=30, min_line_len=15, max_line_gap=5)

    # Combine lines with original image
    result = cv2.addWeighted(image, 0.9, line_image, 1, 0)
    return result


name = 'solidWhiteRight.mp4'
# name = 'solidYellowLeft.mp4'
# name = 'challenge.mp4'

output = 'test_videos_output/' + name
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0, 2)
clip = VideoFileClip('test_videos/' + name)

white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)

print('Finished')