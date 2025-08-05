def auto_detect_12_points(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=100, param1=100, param2=30,
                               minRadius=0, maxRadius=0)

    if circles is None:
        print("No circles detected.")
        sys.exit(1)

    circles = np.uint16(np.around(circles))
    # Use the largest detected circle (you can enhance selection logic)
    c = circles[0][0]
    Cx, Cy, R = c[0], c[1], c[2]

    # Generate 12 evenly spaced points on the circle boundary
    points = []
    for i in range(12):
        theta = 2 * np.pi * i / 12
        x = int(Cx + R * np.cos(theta))
        y = int(Cy + R * np.sin(theta))
        points.append((x, y))
        cv2.circle(input_img, (x, y), 5, (0, 0, 255), -1)

    print(f"Auto-detected circle: Center=({Cx}, {Cy}), Radius={R}")
    cv2.imshow("Auto 12 Points", input_img)
    cv2.waitKey(0)

    return points








#########
def midpoint_circle_warp(input_img):
    h, w = input_img.shape[:2]
    temp_img = input_img.copy()
    cv2.imshow("Input Frame", temp_img)
    cv2.setMouseCallback("Input Frame", mouse_callback)

    print("Mark 12 points on circular boundary")
    while len(g_clicks) < nNumPoints:
        for pt in g_clicks:
            cv2.drawMarker(temp_img, pt, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.imshow("Input Frame", temp_img)
        if cv2.waitKey(100) == 27:  # ESC to exit early
            break

    cv2.setMouseCallback("Input Frame", lambda *args: None)
    if len(g_clicks) < nNumPoints:
        print("Not enough points marked.")
        sys.exit(1)

    points = np.array(g_clicks, dtype=np.float32)
    mean = point_mean(points)
    xi = points[:, 0] - mean[0]
    yi = points[:, 1] - mean[1]

    A = np.vstack([
        2 * xi, 2 * yi, np.ones_like(xi)
    ]).T
    B = (xi**2 + yi**2).reshape(-1, 1)

    res = np.linalg.lstsq(A, B, rcond=None)[0]
    Cx = int(res[0] + mean[0])
    Cy = int(res[1] + mean[1])
    R = int(np.sqrt(res[2] + res[0]**2 + res[1]**2))

    print(f"Estimated Circle Center: ({Cx}, {Cy}), Radius: {R}")

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for v in range(h):
        for u in range(w):
            xt = u - Cx
            yt = v - Cy
            r2 = xt**2 + yt**2
            if r2 == 0:
                map_x[v, u] = u
                map_y[v, u] = v
                continue

            AO1 = (xt**2 + R**2) / (2.0 * xt) if xt != 0 else 1e-6
            AB = np.sqrt(xt**2 + R**2)
            PE = R - yt if (R - yt) != 0 else 1e-6
            a = yt / PE
            b = 2.0 * np.arcsin(np.clip(AB / (2.0 * AO1), -1.0, 1.0))

            alpha = (a * b) / (a + 1.0) if (a + 1.0) != 0 else 0
            x1 = xt - AO1 + AO1 * np.cos(alpha)
            y1 = AO1 * np.sin(alpha)

            map_x[v, u] = x1 + Cx
            map_y[v, u] = y1 + Cy

    # Remap Y-axis for uniformity
    for u in range(w):
        y_col = map_y[:, u]
        min_y, max_y = np.min(y_col), np.max(y_col)
        range_y = max_y - min_y
        scale = h / range_y if range_y > 0 else 1
        map_y[:, u] = (map_y[:, u] - Cy) * scale + Cy

    return map_x, map_y






#####*
import cv2
import numpy as np
import sys
import math

# Methods
HEMICYLINDER = 0
MIDPOINTCIRCLE = 1

# Distortion models
NODISTORTION = -1
EQUIDISTANT = 0
EQUISOLID = 1

g_clicks = []
nNumPoints = 12

def print_help():
    print("ImageWarping Usage:")
    print("python image_warping.py inputimagefile outputimagefile [method]")
    print("Methods:")
    print(" 0 - Hemicylinder (default)")
    print(" 1 - Midpoint circle (needs 12 points)")

def point_mean(points):
    return np.mean(points, axis=0).astype(int)

def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(g_clicks) < nNumPoints:
        g_clicks.append((x, y))
        print(f"Point {len(g_clicks)}: ({x}, {y})")

def hemi_cylinder_warp(input_img, distortion_model=EQUIDISTANT):
    h, w = input_img.shape[:2]
    Cx, Cy = w // 2, h // 2
    F = w / np.pi
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for v in range(h):
        for u in range(w):
            xt = u
            yt = v - Cy
            r = w / np.pi
            alpha = (w - xt) / r
            xp = r * np.cos(alpha)
            yp = yt
            zp = r * abs(np.sin(alpha))
            rp = np.sqrt(xp**2 + yp**2)
            theta = np.arctan2(rp, zp)

            if distortion_model == EQUIDISTANT:
                x1 = F * theta * xp / rp
                y1 = F * theta * yp / rp
            elif distortion_model == EQUISOLID:
                x1 = 2 * F * np.sin(theta / 2) * xp / rp
                y1 = 2 * F * np.sin(theta / 2) * yp / rp
            else:
                x1 = xt
                y1 = yt

            map_x[v, u] = x1 + Cx
            map_y[v, u] = y1 + Cy

    return map_x, map_y

def midpoint_circle_warp(input_img):
    h, w = input_img.shape[:2]
    temp_img = input_img.copy()
    cv2.imshow("Input Frame", temp_img)
    cv2.setMouseCallback("Input Frame", mouse_callback)

    print("Mark 12 points on circular boundary")
    while len(g_clicks) < nNumPoints:
        cv2.waitKey(100)
        for i, pt in enumerate(g_clicks):
            cv2.drawMarker(temp_img, pt, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.imshow("Input Frame", temp_img)

    cv2.setMouseCallback("Input Frame", lambda *args: None)

    points = np.array(g_clicks)
    mean = point_mean(points)
    xi = points[:, 0] - mean[0]
    yi = points[:, 1] - mean[1]

    A = np.vstack([
        2 * xi, 2 * yi, np.ones_like(xi)
    ]).T
    B = (xi**2 + yi**2).reshape(-1, 1)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    Cx, Cy = int(res[0]) + mean[0], int(res[1]) + mean[1]
    R = int(np.sqrt(res[2] + res[0]**2 + res[1]**2))

    print(f"Cx = {Cx}, Cy = {Cy}, R = {R}")

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for u in range(w):
        for v in range(h):
            xt = u - Cx
            yt = v - Cy
            if xt != 0:
                AO1 = (xt**2 + R**2) / (2.0 * xt)
                AB = np.sqrt(xt**2 + R**2)
                PE = R - yt
                a = yt / PE if PE != 0 else 1
                b = 2 * np.arcsin(AB / (2 * AO1))
                alpha = a * b / (a + 1) if a + 1 != 0 else 0
                x1 = xt - AO1 + AO1 * np.cos(alpha)
                y1 = AO1 * np.sin(alpha)
            else:
                x1, y1 = xt, yt

            map_x[v, u] = x1 + Cx
            map_y[v, u] = y1 + Cy

    # Rescale y transform
    for u in range(w):
        col_y = map_y[:, u]
        min_y1, max_y1 = np.min(col_y), np.max(col_y)
        range_y = max_y1 - min_y1
        factor = range_y / h if h != 0 else 1
        factor = factor ** 1.0 if factor > 1.0 else factor
        for v in range(h):
            map_x[v, u] = u
            map_y[v, u] = (v - Cy) / factor + Cy

    return map_x, map_y

def main():
    if len(sys.argv) < 3:
        print_help()
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    method = int(sys.argv[3]) if len(sys.argv) > 3 else HEMICYLINDER

    input_img = cv2.imread(input_file)
    if input_img is None:
        print("Error loading input image.")
        sys.exit(1)

    if method == HEMICYLINDER:
        map_x, map_y = hemi_cylinder_warp(input_img)
    elif method == MIDPOINTCIRCLE:
        map_x, map_y = midpoint_circle_warp(input_img)
    else:
        print_help()
        sys.exit(1)

    output_img = cv2.remap(input_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(output_file, output_img)
    print(f"Output saved to {output_file}")
    cv2.imshow("Output", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








import cv2
import numpy as np

def fisheye_distortion_rect(patch):
    h, w = patch.shape[:2]
    cx, cy = w / 2, h / 2  # center

    # Normalized coordinates in range [-1, 1]
    xv = np.linspace(-1, 1, w)
    yv = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xv, yv)

    # Rescale to make x and y aspect-ratio aware
    scale_x = max(w, h) / w
    scale_y = max(w, h) / h
    xn = xv * scale_x
    yn = yv * scale_y

    # Square to circular mapping
    denom = np.sqrt(1 - (yn**2)/2)
    denom[denom == 0] = 1e-6
    x1 = xn / denom

    denom = np.sqrt(1 - (xn**2)/2)
    denom[denom == 0] = 1e-6
    y1 = yn / denom

    # Apply radial exponential barrel distortion
    r = np.sqrt(x1**2 + y1**2)
    factor = np.exp(-r**2 / 4)
    x2 = x1 * factor
    y2 = y1 * factor

    # Map to pixel coordinates
    map_x = ((x2 / scale_x + 1) * cx).astype(np.float32)
    map_y = ((y2 / scale_y + 1) * cy).astype(np.float32)

    # Remap
    distorted = cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return distorted

# Example usage
img = cv2.imread("patch_rect.jpg")  # Rectangular patch
img = cv2.resize(img, (320, 240))  # Example size

distorted = fisheye_distortion_rect(img)

cv2.imshow("Original", img)
cv2.imshow("Fisheye Distorted", distorted)
cv2.waitKey(0)
cv2.destroyAllWindows()





test: https://drive.google.com/drive/folders/1iEm2qu8gCQwlVnfvNexRzWrstq8DhYLn


# temp

https://drive.google.com/drive/folders/13aU5TKdS7NOLMhEEftDOURCJrSOG7Jf2

To visualize a segmented image of size (5, 320, 320) using OpenCV, you can use the cv2.imshow() function. However, this function expects the image to be in the format (height, width, channels), and it can only display images with 1 (grayscale) or 3 (BGR) channels.

If your segmented image has 5 channels, you might want to visualize each channel separately as a grayscale image. Here’s an example code snippet:
:calibration:
https://drive.google.com/drive/folders/1aO1rSm_30E_LKMh_nfNHvw9reS3lJkEo?usp=sharing





calib_apr:

https://drive.google.com/drive/folders/1LDJel7S2ChMmaynHoH2cRCqzTedDbfxd?usp=sharing







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
link: https://drive.google.com/drive/folders/18SPoUREVL3bV6iab8uYesrI3yqpcmkgE?usp=sharing

https://drive.google.com/drive/folders/1QBUBeTPfe5KaW2STvEnOywW9toz7aqFv?usp=drive_link


https://drive.google.com/drive/folders/1JQNov3Cu6Uv6EFAra8Lv-EDgaq-6NBoQ
bag_path = "path_to_your_ros2_bag"
topic_name = "/your_pointcloud_topic"
points = read_point_cloud_from_ros2_bag(bag_path, topic_name)
save_xyzi_to_pcd("output_xyzi.pcd", points)
blender: https://drive.google.com/drive/folders/1RXv5qIzaN6hYvGvPoCMmF_Ij7eMVx2nO?usp=sharing
