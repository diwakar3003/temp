import cv2
import numpy as np

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using the given camera matrix and distortion coefficients."""

    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return undistorted_image

# Load the image and camera parameters
image = cv2.imread('/home/diwakar/Desktop/tutorial_distortion_calibration/images/cam_left.png')

# Camera matrix and distortion coefficients
camera_matrix =np.array([[658.98431343,0.,302.50047925],[0.,659.73448089,243.58638325],[0.,0.,1.]])
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Replace with your camera matrix
dist_coeffs = np.array([-2.47543410e-01,7.14418145e-02,-1.83688275e-04,-2.74411144e-04,1.05769974e-01])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Replace with your distortion coefficients

# Undistort the image
undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()