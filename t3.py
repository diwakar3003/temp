import cv2
import numpy as np

# Load camera matrix and distortion coefficients
mtx =np.array([[658.98431343,0.,302.50047925],[0.,659.73448089,243.58638325],[0.,0.,1.]])

dist =np.array([-2.47543410e-01,7.14418145e-02,-1.83688275e-04,-2.74411144e-04,1.05769974e-01])

# Read an image
img = cv2.imread('/home/diwakar/Desktop/tutorial_distortion_calibration/images/cam_left.png')

# Compute undistortion maps
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# # undistort
#dst1 = cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist,newCameraMatrix=newcameramtx)
# #cv2.imwrite('calibresult.png', dst1)
# cv2.imshow('chess board', np.hstack((img, dst1)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Crop the undistorted image
# x, y, w, h = roi
# dst1_croped = dst1 [y:y + h, x:x + w]
# #dst2 = dst1 [y:y+h, x:x+w]
# cv2.imshow('chess board1', np.hstack((img, dst1)))
# cv2.imshow('chess board', dst1_croped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# cv2.imshow('chess board', dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# crop the image
x, y, w, h = roi
dst2_croped = dst2[y:y+h, x:x+w]
cv2.imshow('chess board1', dst2)
cv2.imshow('chess board', dst2_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

# # Undistort the image
# # undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# # Show the original and undistorted images
# cv2.imshow('Original Image', img)
# cv2.imshow('Undistorted Image', undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()