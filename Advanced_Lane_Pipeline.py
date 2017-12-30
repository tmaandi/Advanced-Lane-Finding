import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# %matplotlib inline
#%matplotlib qt
import glob

#############################
####### Loading Images ######
#############################

test_images_names = ['straight_lines1.jpg','straight_lines2.jpg','test1.jpg','test2.jpg','test3.jpg',\
                    'test4.jpg','test5.jpg','test6.jpg'] 
test_images_dir = './test_images/'

test_images = []

for image_name in test_images_names:
    image = mpimg.imread(test_images_dir + image_name)
    test_images.append(image)

test_images = np.array(test_images)

cal_images_dir = './camera_cal/'
calibration_images = []

for image_name in glob.glob(cal_images_dir + 'calibration*.jpg'):
    im = mpimg.imread(image_name)
    calibration_images.append(im)

calibration_images = np.array(calibration_images)

##############################
### Camera Calibration #######
##############################

# Prepare object points
nx = 9 # number of inside corners in x
ny = 6 # number of inside corners in y

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....(8,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

for cal_image in calibration_images:
    gray_cal_image = cv2.cvtColor(cal_image,cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_cal_image, (nx, ny), None)

    # If found, draw corners, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        #plt.figure()
        #cv2.drawChessboardCorners(cal_image, (nx, ny), corners, ret)
        #plt.imshow(cal_image)
        #plt.show()
    else:
        #plt.figure()
        #print('Corners not identified for this image')
        #plt.imshow(cal_image)
        #plt.show()
        pass

# Camera Calibration Matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,\
                                gray_cal_image.shape[::-1], None, None)

# Undisorting a test image(s)
test_img1 = mpimg.imread(cal_images_dir + 'test1.jpg')
dst1 = cv2.undistort(test_img1, mtx, dist, None, mtx)

test_img2 = mpimg.imread(cal_images_dir + 'test2.jpg')
dst2 = cv2.undistort(test_img2, mtx, dist, None, mtx)
"""
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,12))
f.tight_layout()
ax1.imshow(test_img1)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(dst1)
ax2.set_title('Undistorted Image', fontsize=25)
ax3.imshow(test_img2)
ax3.set_title('Original Image', fontsize=25)
ax4.imshow(dst2)
ax4.set_title('Undistorted Image', fontsize=25)
"""
# Perspective Transform

def outermostCorners(corners,nx,ny):
    # corners = 54 x 1 x 2
    p1 = [corners[0,0,0],corners[0,0,1]]
    p2 = [corners[nx-1,0,0],corners[nx-1,0,1]]
    p3 = [corners[nx*ny-1,0,0],corners[nx*ny-1,0,1]]
    p4 = [corners[nx*ny-nx,0,0],corners[nx*ny-nx,0,1]]
    outercorners = [p1,p2,p3,p4]
    result = np.float32(outercorners)
    return result

def corners_unwarp(img, nx, ny, mtx, dist):

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
#     plt.imshow(undistorted)
#     plt.show()
    gray_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
#     plt.imshow(gray_img)
#     plt.show()
    ret,corners = cv2.findChessboardCorners(gray_img, (nx,ny),None)
    img_size = gray_img.shape[::-1]
    offset = 100
#     print(ret)
    if ret == True:
        cv2.drawChessboardCorners(undistorted, (nx,ny),corners,ret)
        src = outermostCorners(corners,nx,ny)
        dest = np.float32([[offset,offset],[img_size[0]-offset,offset],\
        [img_size[0]-offset,img_size[1]-offset],[offset,img_size[1]-offset]])
    else:
        print('Corners not identified for this image')
    
    M = cv2.getPerspectiveTransform(src,dest)
    warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M
"""

# Testing perspective and distortion correction
perspective_test_img = calibration_images[11]
top_down, perspective_M = corners_unwarp(perspective_test_img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
ax1.imshow(perspective_test_img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

"""




########################################
### Gradient and Color Thresholding ####
########################################

img = test_images[1]
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(undistorted_img,cmap='gray')
ax2.set_title('Undistorted', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

### Gradient Binary ###

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
	grad_binary[(np.absolute(grad) > thresh[0]) & (np.absolute(grad)<=thresh[1])] = 1

	return grad_binary

def mag_thresh(img, sobel_kernel=3,mag_thresh=(0,255)):
	gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	gradX = cv2.Sobel(gray_img, cv2.CV_64F,1,0,ksize = sobel_kernel)
	gradY = cv2.Sobel(gray_img, cv2.CV_64F,0,1,ksize = sobel_kernel)
	gradMag = np.sqrt(np.square(gradX) + np.square(gradY))
	gradMag = (gradMag/np.max(gradMag))*255
	gradMag_binary = np.zeros_like(gradMag)
	gradMag_binary[(gradMag > mag_thresh[0]) & (gradMag <= mag_thresh[1])] = 1
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

# Apply each of the thresholding functions
gradX = abs_sobel_thresh(undistorted_img, orient='x', sobel_kernel = ksize, thresh=(40,255))
gradY = abs_sobel_thresh(undistorted_img, orient='y', sobel_kernel = ksize, thresh=(40,255))
mag_binary = mag_thresh(undistorted_img, sobel_kernel=ksize, mag_thresh=(40,255))
dir_binary = dir_threshold(undistorted_img, sobel_kernel=ksize, thresh=(0.6, 1.3))

combined = np.zeros_like(dir_binary)
# combined[((gradX == 1) & (gradY == 1))| ((mag_binary == 1) & (dir_binary == 1))] = 1
combined[((mag_binary == 1) & (dir_binary == 1))] = 1
#combined[((gradX == 1) & (gradY == 1))] = 1

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(undistorted_img)
ax1.set_title('Undistorted Image', fontsize=25)
ax2.imshow(combined, cmap='gray')
ax2.set_title(' Grad. Mag. and Dir Thresholding', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

### Color Binary ###

def hls_select(img, thresh=(0, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output
    
hls_binary = hls_select(undistorted_img, thresh=(190, 255))
hls_graddir_binary = np.zeros_like(hls_binary)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(undistorted_img)
ax1.set_title('Undistorted Image', fontsize=25)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('"S" Thresholding', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

### Combined color and gradient thresholding ###

color_grad_thresh = np.zeros_like(combined)
color_grad_thresh_stacked = np.uint8(np.dstack((np.zeros_like(combined), combined, hls_binary))*255)

# combining grad mag and dir thresh with "S" thresh
color_grad_thresh[(combined == 1)|(hls_binary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.tight_layout()
ax1.imshow(color_grad_thresh_stacked)
ax1.set_title('Stacked thresholds',fontsize=25)
ax2.imshow(color_grad_thresh, cmap='gray')
ax2.set_title('Combined S channel and gradient thresholds',fontsize=25)
plt.show()

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Masking
masking_pts = np.int32(np.array([[[50,720],[580,450],[color_grad_thresh.shape[1]-520,450],[color_grad_thresh.shape[1]-50,720]]]))
masked_img = region_of_interest(color_grad_thresh, masking_pts)

# Drawing mask on input image
masked_annot_undistorted = np.copy(undistorted_img)
cv2.polylines(masked_annot_undistorted,masking_pts,True,(255,0,0), 5)

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.tight_layout()
ax1.imshow(masked_annot_undistorted,  cmap='gray')
ax1.set_title('Mask-Annotated Undistorted Image',fontsize=25)
ax2.imshow(masked_img, cmap='gray')
ax2.set_title('Masked Image',fontsize=25)
plt.show()

########################################
######### Perpective Transform #########
########################################

def image_warp(src,dst,img):
    # Here, img should be undistorted
#     src = src.tolist()
#     dst = dst.tolist()
    src = np.float32(src)
    dst = np.float32(dst)
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M

def draw_lines(img, pts, color = (255,0,0)):
   
    cv2.polylines(img,[pts],True,color, 5)
    
src_pts = np.array([[278,675],[552,480],[734,480],[1040,675]]) # based on test_images[1]
dst_pts = np.array([[418,720],[418,0],[900,0],[900,720]])

annotated_img = np.copy(undistorted_img)

draw_lines(annotated_img,src_pts)
draw_lines(annotated_img,dst_pts,(0,0,255))

warped_img, M = image_warp(src_pts, dst_pts,masked_img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(annotated_img)
ax1.set_title('Perspective-Annotated Image', fontsize=25)
ax2.imshow(warped_img, cmap = 'gray')
ax2.set_title('Warped Image', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

########################################
############ Finding Lanes #############
########################################

histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
plt.show()
# Create an output image to draw on and  visualize the result
out_img = np.dstack((warped_img, warped_img, warped_img))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(warped_img.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped_img.shape[0] - (window+1)*window_height
    win_y_high = warped_img.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

   