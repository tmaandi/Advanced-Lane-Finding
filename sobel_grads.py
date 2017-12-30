import numpy as np 
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def abs_sobel_thresh(img,orient='x',sobel_kernel=3, thresh=(0,255)):

	gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	if orient== 'x':
		grad = cv2.Sobel(gray_img, cv2.CV_64F,1,0,ksize = sobel_kernel)
		grad = (grad/np.max(np.absolute(grad)))*255
	elif orient=='y':
		grad = cv2.Sobel(gray_img, cv2.CV_64F,0,1,ksize = sobel_kernel)
		grad = (grad/np.max(np.absolute(grad)))*255
	else:
		raise("Incorrect orientation choice: choose 'x' or 'y'")

	grad_binary = np.zeros_like(grad)
	grad_binary[(np.absolute(grad) >= thresh[0]) & (np.absolute(grad)<=thresh[1])] = 1

	return grad_binary

def mag_thresh(img, sobel_kernel=3,mag_thresh=(0,255)):
	gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	gradX = cv2.Sobel(gray_img, cv2.CV_64F,1,0,ksize = sobel_kernel)
	gradY = cv2.Sobel(gray_img, cv2.CV_64F,0,1,ksize = sobel_kernel)
	gradMag = np.sqrt(np.square(gradX) + np.square(gradY))
	gradMag = (gradMag/np.max(gradMag))*255
	gradMag_binary = np.zeros_like(gradMag)
	gradMag_binary[(gradMag >= mag_thresh[0]) & (gradMag <= mag_thresh[1])] = 1
	return gradMag_binary

def dir_threshold(img,sobel_kernel = 3, thresh=(0, np.pi/2)):
	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gradX = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize = sobel_kernel)
	gradY = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize = sobel_kernel)
	dir = np.arctan2(np.absolute(gradY),np.absolute(gradX))
	dir_binary = np.zeros_like(dir)
	dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
	return dir_binary


# Choose a Sobel kernel size
ksize = 15

image = mpimg.imread("./images/signs_vehicles_xygrad.png")

# Apply each of the thresholding functions
gradX = abs_sobel_thresh(image, orient='x', sobel_kernel = ksize, thresh=(0,255))
gradY = abs_sobel_thresh(image, orient='y', sobel_kernel = ksize, thresh=(0,255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50,255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.2))

combined = np.zeros_like(dir_binary)
combined[((gradX == 1) & (gradY == 1))| ((mag_binary == 1) & (dir_binary == 1))] = 1

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
# ax2.imshow(dir_binary, cmap='gray')
ax2.imshow(combined, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=25)
ax2.set_title('Combined Thresholding', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()