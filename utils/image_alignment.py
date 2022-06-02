"""
-- Created by: Netra Prasad Neupane
-- Created on: 5/16/22
"""

from scipy import ndimage
import cv2
import numpy as np
import math

def align_image(image):
    """
    Probabilistic Hough Line transform with average slope of lines
    """

    angles = []
    img_copy = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    lines = cv2.HoughLinesP(threshed_image, 1, np.pi / 180, 200, None, 150, 10)
    # Get the angle from the line
    if lines is not None:
        horizontal_lines = []
        for i, line in enumerate(lines):
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            diff_x = x2 - x1
            diff_y = y2 - y1
            if abs(diff_y) < 30 and abs(diff_x) > 0:
                horizontal_lines.append((x1, y1, x2, y2))
                try:
                    slope = diff_y / diff_x
                    angle = math.degrees(math.atan(slope))
                    angles.append(angle)
                except Exception as e:
                    print(e)
                    continue

        if len(angles) > 0:
            rotation_angle = sum(angles) / len(angles)
        else:
            rotation_angle = 0
        # Rotate the image
        img_rotated = ndimage.rotate(image, rotation_angle, reshape=True)
        return img_rotated

    else:
        return img_copy