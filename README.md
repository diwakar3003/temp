# temp

To visualize a segmented image of size (5, 320, 320) using OpenCV, you can use the cv2.imshow() function. However, this function expects the image to be in the format (height, width, channels), and it can only display images with 1 (grayscale) or 3 (BGR) channels.

If your segmented image has 5 channels, you might want to visualize each channel separately as a grayscale image. Here’s an example code snippet:
:calibration:
https://drive.google.com/drive/folders/1aO1rSm_30E_LKMh_nfNHvw9reS3lJkEo?usp=sharing







import cv2
import numpy as np

# Assuming 'output' is your model's output array
output = ...

# The output shape should be [5, 320, 320]
assert output.shape == (5, 320, 320)

# For each channel in the image
for i in range(output.shape[0]):
    # Normalize the channel to 0-255
    normalized = cv2.normalize(output[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Display the image
    cv2.imshow('Segmented Image - Channel {}'.format(i+1), normalized)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
Copy
This code will create a separate window for each channel of the segmented image. Note that this code uses OpenCV’s normalization function to scale the pixel values to the range 0-255, which is necessary for cv2.imshow() to display the images correctly. If your pixel values are already in this range, you can skip this step.

#########
import cv2
import numpy as np

# Assuming 'output' is your model's output array
output = ...

# The output shape should be [5, 320, 320]
assert output.shape == (5, 320, 320)

# For each channel in the image
for i in range(output.shape[0]):
    # Normalize the channel to 0-255
    normalized = cv2.normalize(output[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Display the image
    cv2.imshow('Segmented Image - Channel {}'.format(i+1), normalized)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
link for data: [[https://drive.google.com/drive/folders/1kasMNp5iehd5kTx-k7ZoudbD57nzX98K?usp=sharing](https://drive.google.com/drive/folders/1ekuchvUT4xmFHFAmOmeLHHCWydJv0chx?usp=sharing)]

(https://drive.google.com/drive/folders/13aU5TKdS7NOLMhEEftDOURCJrSOG7Jf2?usp=sharing)

https://drive.google.com/drive/folders/1V9uKck3Ec9jlwfgaxcX0NjNubQDc11Hi
https://drive.google.com/drive/folders/1DSax4ccFzPgJ1peZGzmfR6QrMwlLlQfZ


import numpy as np
import rosbag2_py
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

def read_point_cloud_from_ros2_bag(bag_path, topic_name):
    """
    Read point cloud data from a ROS 2 bag file.

    Parameters:
    - bag_path: Path to the ROS 2 bag file.
    - topic_name: The topic name containing the PointCloud2 data.

    Returns:
    - A numpy array with shape (N, 4) where each row is [x, y, z, intensity].
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Define an empty list to collect point cloud data
    points_list = []

    # Iterate through messages
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic == topic_name:
            # Deserialize the PointCloud2 message
            point_cloud_msg = PointCloud2()
            point_cloud_msg.deserialize(data)

            # Use pc2.read_points to extract x, y, z, intensity fields
            for point in pc2.read_points(point_cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points_list.append(point)

    # Convert the list to a numpy array
    return np.array(points_list)


def save_xyzi_to_pcd(filename, points):
    """
    Save a point cloud with x, y, z, and intensity fields to a PCD file in ASCII format.

    Parameters:
    - filename: The name of the file to save the point cloud.
    - points: A numpy array of shape (N, 4), where each row is [x, y, z, intensity].
    """
    # Ensure points are in the correct shape
    assert points.shape[1] == 4, "Input array must have shape (N, 4) for x, y, z, intensity."

    # Define the PCD header for ASCII format
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""

    # Write header and point data to the file
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%f %f %f %f')
    
    print(f"Point cloud saved as {filename}")


# Example usage
bag_path = "path_to_your_ros2_bag"
topic_name = "/your_pointcloud_topic"
points = read_point_cloud_from_ros2_bag(bag_path, topic_name)
save_xyzi_to_pcd("output_xyzi.pcd", points)
