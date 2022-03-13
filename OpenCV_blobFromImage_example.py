from imutils import paths
import numpy as np
import cv2
import os


chdir = os.getcwd()
# Load the class labels from the disks
rows = open("./resources/synset_words.txt").read().strip().split("\n")
# To get first label from each row in given clases
classes = [r[r.find(" ")+1:].split(",")[0] for r in rows]

# Load our serialized model from disk
# specify bvlc_googlenet.prototxt as the filename parameter and
# bvlc_googlenet.caffemodel as the actual model file

config_path = os.path.join(chdir,'resources/bvlc_googlenet.prototxt')
model_path = os.path.join(chdir,'resources/bvlc_googlenet.caffemodel')
net = cv2.dnn.readNetFromCaffe(config_path,model_path)


# grab the paths to the input images
image_paths = sorted(list(paths.list_images("images/")))

# (1) load the first image from disk, (2) pre-process it by resizing
# it to 224x224 pixels, and (3) construct a blob that can be passed
# through the pre-trained network
image = cv2.imread(image_paths[0])
resized = cv2.resize(image,(224,224))
blob = cv2.dnn.blobFromImage(resized,1,(224,224),(104,117,123))
print('First blob :{}'.format(blob.shape))


# set the input to the pre-trained deep learning network and obtain
# the output predicted probabilities for each of the 1,000 ImageNet
# classes
net.setInput(blob)
preds = net.forward()

# sort the probabilities (in descending) order, grab the index of the
# top predicted label, and draw it on the input image
idx = np.argsort(preds[0])[::-1][0]
# print(classes[idx])
text = "Label: {}, {:2f}%".format(classes[idx],preds[0][idx]*100)
cv2.putText(image,text,(5,25),cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,255,0),2)
# Show the output image
cv2.imshow("Image",image)
cv2.waitKey(0)


# For n Images
# initialize the list of images we'll be passing through the network
images = []
# loop over the input images (excluding the first one since we
# already classified it), pre-process each image, and update the
# `images` list
for p in image_paths[1:]:
	image = cv2.imread(p)
	image = cv2.resize(image, (224, 224))
	images.append(image)
# convert the images list into an OpenCV-compatible blob
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# set the input to our pre-trained network and obtain the output
# class label predictions
net.setInput(blob)
preds = net.forward()
# loop over the input images
for (i, p) in enumerate(image_paths[1:]):
	# load the image from disk
	image = cv2.imread(p)
	# find the top class label from the `preds` list and draw it on
	# the image
	idx = np.argsort(preds[i])[::-1][0]
	text = "Label: {}, {:.2f}%".format(classes[idx],
		preds[i][idx] * 100)
	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255,0), 2)
	# display the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
cv2.destroyAllWindows()