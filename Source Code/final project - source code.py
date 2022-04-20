"""
Otsu's Algorithm
"""

from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

#creating an empty dictionary to use later, 
#dictionaries are used to store date values
threshold_values = {}
#creating a list size 1
h = [1]

#creating a function to calculate the histogram of an image
def histogram(img):
   #finding the shape of the image
   row, col = img.shape 
   #creating a new array filled with zeros
   y = np.zeros(256)
   #creating a nested for loop where the outer loop executes 
   #only once when its test expression is true
   for i in range(0, row):
      for j in range(0, col):
         #the operator += adds the right operand to the left operand 
         #and assigns the result to the left operand
         y[img[i,j]] += 1
   #returns evenly spaced values within a given interval
   x = np.arange(0,256)
   #to plot the generated histogram
   plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   #to show the generated histogram
   plt.show()
   return y

#to generate the output image after performing 
#the Otsu's algorithm to input histogram
def generate_output_img(img, threshold):
    #to find out the shape the image
    row, col = img.shape 
    #create an array filled with zeros the size of the input image
    y = np.zeros((row, col))
    #using a nested for loop to generate an image with only black and white pixels
    for i in range(0,row):
        for j in range(0,col):
            #if else statment to determine if the pixel will be black or white
            if img[i,j] >= threshold:
                #value of white pixel
                y[i,j] = 255
            else:
                #value of black pixel
                y[i,j] = 0
    return y

#function to calculate weight
def weight(p, q):
    #defining initial variable
    w = 0
    #for loop 
    for i in range(p, q):
        #the operator += adds right operand to left operand and assigns the result to the left operand
        w += h[i]
    return w

#function to calculate the mean
def mean(p, q):
    #defining initial variable
    m = 0
    #calling on the weight function previously created
    w = weight(p, q)
    #generating a for loop
    for i in range(p, q):
        #used to add right and left side and assign the results to the left side
        m += h[i] * i  
    return m/float(w)

#function to calculate the variance
def variance(p, q):
    #defining initial variable
    v = 0
    #calling on mean function
    m = mean(p, q)
    #calling on weight function
    w = weight(p, q)
    #for loop
    for i in range(p, q):
        #equation to calculate the variance
        v += ((i - m) **2) * h[i]
    #operator /= divides left operand with right operand and assigns result to left operand
    v /= w
    return v        

#function to find out the number of pixels
def total_pixels(h):
    #defining initial variable
    num_pixels = 0
    #for loop
    for i in range(0, len(h)):
        #if statement
        if h[i]>0:
           #if the values of h[i] are greater than 0, will be added to the initial variable
            num_pixels += h[i]
    return num_pixels

#creating Otsu's threshold function
def otsu_threshold(input_histogram):
    #calling previous function to know quantity of pixels as defined in a histogram
    num_pixels = total_pixels(input_histogram)
    #for loop from 1 to the length of the input histogram
    for i in range(1, len(input_histogram)):
        #variance, background
        vb = variance(0, i)
        #weight, background
        wb = weight(0, i) / float(num_pixels)
        
        #variance, foreground
        vf = variance(i, len(input_histogram))
        #weight, foreground
        wf = weight(i, len(input_histogram)) / float(num_pixels)
        
        #within class variance
        wcv = wb * (vb) + wf * (vf)
        
        #the keywork "not" is used to invert an expression, so if it returns False, 
        #it is now True. here i want to invert the value of the boolean variable.
        #.isnan checks whether a value is NaN (Not a Number), or not. this method 
        #returns True if the specified value is a Nan, otherwise it returns False.
        if not math.isnan(wcv):
            #creating a list containing the thresholding values
            threshold_values[i] = wcv       
    return

#to figure out what the optimal thresholding limit should be
def calculate_optimal_threshold():
    #to find the minimum of all the values of the previously calculated threshhold values
    min_wcv = min(threshold_values.values())
    #this is a dictionary where it will show the optimal threshold value in the console
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_wcv]
    #display the text in the console next to the optimal threshold
    print ('the optimal threshold is: ', optimal_threshold[0])
    return optimal_threshold[0]

#this was just using a premade module to display the histogram so that i could 
#compare it with the one generated with the code
img = cv2.imread('C:/Users/Berto/test.PNG', 0)
plt.subplot(2,1,1), plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2), plt.hist(img.ravel(), 256)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
plt.show()

#opening an image and convering the image into an array
image = Image.open('C:/Users/Berto/test.PNG').convert("L")
img = np.asarray(image)

#running the histogram function
h = histogram(img)
#running the Otsu algorithm function
otsu_threshold(h)
#running the program to find the optimal thresholding value
optimal_thresholding_value = calculate_optimal_threshold()
#getting the output image 
output_image = generate_output_img(img, optimal_thresholding_value)
#plotting the output image
plt.imshow(output_image)
#saving the output image
cv2.imwrite('otsu output.jpg', output_image)

"""
Watershed method
"""

import numpy as np
import cv2

#import the image
img = cv2.imread('C:/Users/Berto/test.PNG')

#even though input image does 'look' grey, Python still considers it an RGB, 
#so use cv2.cvtColor to make Python register it as a grayscale image
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#the cv2.threshold module is used to apply the thresholding to a greyscale image
ret, thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#creating a kernal that will be used later to help remove noise
noise_kernel = np.ones((3,3), np.uint8)

#close operation to fill holes
closing_small_holes = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, noise_kernel, iterations = 1)

#finding the sure background area
sure_background = cv2.dilate(closing_small_holes, noise_kernel, iterations = 1)

#finding the sure foreground area
dist_transform = cv2.distanceTransform(sure_background, cv2.DIST_L2, 3)
ret, sure_foreground = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

#finding the unknown area
sure_foreground1 = np.uint8(sure_foreground)
unknown_region = cv2.subtract(sure_background,sure_foreground1)

#marker labelling, required for the cv2.watershed algorithm later
ret, marker = cv2.connectedComponents(sure_foreground1)
# Add one to all labels so that sure background is not 0, but 1
marker = marker+1
#mark the unkonw region with zero
marker[unknown_region==255] = 0

#run the cv2.watershed function
marker = cv2.watershed(img,marker)
img[marker == -1] = [255, 0, 0]
img1 = img

#run the cv2.erode module to show the actual foreground
foreground = cv2.erode(thresh, None, iterations = 1)
cv2.imshow('foreground of image', foreground)

#run the cv2.dilate module to show the actual background
background = cv2.dilate(thresh, None, iterations = 1)
ret, background1 = cv2.threshold(background, 1, 175, 1)
cv2.imshow('background', background1)

#run the cv2.add module to combine the foregound and background images
marker = cv2.add(foreground, background1)

#convert the image into having integer values
marker32 = np.int32(marker)

#run the cv2.watershed function
new_marker = cv2.watershed(img1,marker32)

#cv2.convertScaleAbs scales, calculates absolute values, and converts the result to 8-bit
m = cv2.convertScaleAbs(new_marker)

#the cv2.threshold module is used to apply the thresholding to m
ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#the cv2.bitwise_and module calculates the per-element bit-wise logical conjunction for two arrays
res = cv2.bitwise_and(img, img, mask = thresh)
cv2.imshow('res', res)

#saving the images
cv2.imwrite('foreground of image.jpg', foreground)
cv2.imwrite('backgound of image.jpg', background1)
cv2.imwrite('watershed output.jpg', res)
cv2.imwrite('marker.jpg', marker32)

"""
Initial comparison
"""

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

#creating a mean square error function
def mean_square_error(image1, image2):
	#sum of the squared difference between the two images
	error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
	error /= float(image1.shape[0] * image1.shape[1])
	return error

#creating function to compare the mean square error and the structural similarity of two images
def compare_images(image1, image2, title):
	#index for the images
	mse = mean_square_error(image1, image2)
	ss = ssim(image1, image2)
	#setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (mse, ss))
	# how first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(image1, cmap = plt.cm.gray)
	plt.axis("off")
	#show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(image2, cmap = plt.cm.gray)
	plt.axis("off")
	#show the images
	plt.show()
    
#loading the images
original = cv2.imread("C:/Users/Berto/test.PNG")
ws = cv2.imread("C:/Users/Berto/watershed output.jpg")
otsu = cv2.imread("C:/Users/Berto/otsu output.jpg")

#convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
ws = cv2.cvtColor(ws, cv2.COLOR_BGR2GRAY)
otsu = cv2.cvtColor(otsu, cv2.COLOR_BGR2GRAY)

#set up the figure
fig = plt.figure("Images")
images = ("original", original),  ("watershed", ws), ("otsu", otsu)
#for loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()

#compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, ws, "Original vs. watershed")
compare_images(original, otsu, "Original vs. otsu")
compare_images(ws, otsu, "watershed vs. otsu")

"""
Converting from greyscale to black and white image
"""

from PIL import Image, ImageChops 

#import image
input_image = Image.open('C:/Users/Berto/test.PNG')
watershed_image = Image.open('C:/Users/Berto/marker.jpg')

#defining variable to be used later
thresh = 175
#essentially and if statment, which will convert the input image into a black 
#and white image so that it can be compared with a different black and white image
fn = lambda x : 255 if x > thresh else 0

#convert the image into a greyscale so that python can interpret it. 
#the '.point' is so that python can map this image through a lookup function
original = input_image.convert('L').point(fn, mode='1')
black_and_white_watershed = watershed_image.convert('L').point(fn, mode='1')

#since input image for the watershed marker was coming out with a black foreground 
#and a white background, i had to invert it so that i could compare it and have it make sense
black_and_white_watershed = ImageChops.invert(black_and_white_watershed) 

original.save('black and white original.png')
black_and_white_watershed.save('black and white watershed.png')

"""
Final comparison
"""
    
#loading the images
original = cv2.imread("C:/Users/Berto/test.PNG")
ws = cv2.imread("C:/Users/Berto/watershed output.jpg")
otsu = cv2.imread("C:/Users/Berto/otsu output.jpg")
black_and_white_original = cv2.imread("C:/Users/Berto/black and white original.png")
black_and_white_watershed = cv2.imread("C:/Users/Berto/black and white watershed.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
ws = cv2.cvtColor(ws, cv2.COLOR_BGR2GRAY)
otsu = cv2.cvtColor(otsu, cv2.COLOR_BGR2GRAY)
bw_original = cv2.cvtColor(black_and_white_original, cv2.COLOR_BGR2GRAY)
bw_watershed = cv2.cvtColor(black_and_white_watershed, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("watershed", ws), ("otsu", otsu), ("b&w original", bw_original), ("b&w watershed", bw_watershed)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(2, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()

# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, ws, "Original vs. watershed")
compare_images(original, otsu, "Original vs. otsu")
compare_images(ws, otsu, "watershed vs. otsu")
compare_images(bw_original, bw_watershed, "black and white Original vs. black and white watershed")
compare_images(bw_original, otsu, "black and white original vs. otsu")
compare_images(bw_watershed, otsu, "black and white watershed vs. otsu")

#so the IDE does not crash
cv2.waitKey(0)
cv2.destroyAllWindows()
