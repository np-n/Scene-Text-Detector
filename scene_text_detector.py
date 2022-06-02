import cv2
import time
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import numpy as np
from utils.image_alignment import align_image
import argparse


def east_detector(image_path):
    # load the input image and grab the image dimensions
    image = cv2.imread(image_path)
    original_image = image.copy()

    aligned_image = align_image(image)
    plt.imshow(aligned_image)
    aligned_original = aligned_image.copy()

    image = aligned_image

    h,w = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    new_width,new_height = 640,320 # Height and Width must be in scale of 32
    rw = w / float(new_width)
    rh = h / float(new_height)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image,(new_width,new_height))
    h,w = image.shape[:2]
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet('./resources/frozen_east_text_detection.pb')

    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image,1.0,(w,h),(123.60,116.78,103.94),
                                 swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    scores,geometry = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0,num_rows):
        scores_data = scores[0,0,y]
        xdata0 = geometry[0, 0, y]
        xdata1 = geometry[0, 1, y]
        xdata2 = geometry[0, 2, y]
        xdata3 = geometry[0, 3, y]
        angles_data = geometry[0,4,y]

        # loop over the number of columns
        for x in range(0, num_cols):
            # if our score does not have sufficient probability, ignore it
            if scores_data[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xdata0[x] + xdata2[x]
            w = xdata1[x] + xdata3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            end_x = int(offset_x + (cos * xdata1[x]) + (sin * xdata2[x]))
            end_y = int(offset_y - (sin * xdata1[x]) + (cos * xdata2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * rw)
        start_y = int(start_y * rh)
        end_x = int(end_x * rw)
        end_y = int(end_y * rh)

        # draw the bounding box on the image
        cv2.rectangle(aligned_original, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    #cv2.imshow("Text Detection", original_image)
    plt.imshow(aligned_original,'gray')
    plt.show()
    return aligned_original



if __name__ == "__main__":
    # local storage image path
    parser = argparse.ArgumentParser(description='Location of test image')
    parser.add_argument('-i','--input_image',help='test image location',default='./images/scene_image1.jpeg')
    args = parser.parse_args()
    image_path = args.input_image
    image = east_detector(image_path)
    cv2.imwrite('./images/output_scene_image2.jpg',image)


