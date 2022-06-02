## Scene Text Detector

 - EAST: An Efficient and Accurate Scene Text Detector(EAST) is deep learning based text detector.
 - The EAST pipeline is capable of predicting words and lines of text at arbitrary orientations on 720p images, and furthermore, can run at 13 FPS, according to the authors.

### Install Detector
- To install detector, first clone or download the repository
- Install the required dependency using `pip` package manager
```
pip install -r requirements.txt
```

### Run Detector
- To run detector go to the root directory i.e. inside the `Scene-Text-Detector`
- Open the terminal in that directory and enter 
```commandline
python scene_text_detector.py -i images/scene_image2.jpg
```

### Examples
Input Image:

<img src="./images/scene_image2.jpg">

Output Image:

<img src="./images/output_scene_image2.jpg">

