from detecto import core as detecto_core
from detecto import utils as detecto_utils
from detecto import visualize as detecto_visualize
from sklearn.svm import SVC

from google_drive import download_file

from skimage import exposure
from skimage import feature
import urllib.request

import joblib

import cv2
import numpy as np
import imutils
from settings import *
import os


class SVMHOGClassifier:
    # HOG settigns
    _orientations = 15 
    _pixels_per_cell = (16, 16) 
    _cells_per_block = (4, 4) 
    _transform_sqrt = True 
    _block_norm = "L1" 
    _visualize = True 
    hog_settings = [_orientations, _pixels_per_cell, _cells_per_block, _transform_sqrt, _block_norm, _visualize]
    

    def __init__(self, img_height=200, img_width=200, hog_settings=None):
        if hog_settings is not None:
            self.hog_settings = hog_settings
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        if not os.path.exists(SVM_MODEL_FILE_PATH):
            print("Downloading svm weights...")
            download_file(SVM_MODEL_DRIVE_ID, SVM_MODEL_FILE_PATH)
        self.clf = joblib.load(SVM_MODEL_FILE_PATH) 


    def GetHOG (self, img):
        HOG, img_HOG = feature.hog(img, 
                                orientations = self.hog_settings[0], 
                                pixels_per_cell = self.hog_settings[1], 
                                cells_per_block = self.hog_settings[2], 
                                transform_sqrt = self.hog_settings[3], 
                                block_norm = self.hog_settings[4],
                                visualize = self.hog_settings[5])
        return (HOG, img_HOG)


    def classificate(self, logo_img):
        if logo_img.shape[0] == 0 or logo_img.shape[1] == 0:
            return False

        img = cv2.resize(logo_img, (self.IMG_WIDTH, self.IMG_HEIGHT))

        if DEBUG:
            cv2.imshow("image", img) 
            cv2.waitKey()  
            cv2.destroyAllWindows()  

        HOG, HOG_img = self.GetHOG(img.copy())
        prediction = int(self.clf.predict([HOG]))

        return prediction == 2


class RCNNDetector:
    def __init__(self):
        if not os.path.exists(RCNN_MODEL_FILE_PATH):
            print("Downloading rcnn weights...")
            download_file(RCNN_MODEL_DRIVE_ID, RCNN_MODEL_FILE_PATH)
        self.detector = detecto_core.Model.load(RCNN_MODEL_FILE_PATH, ["kruzhok"])

    
    def predict(self, image):
        return self.detector.predict(image)


class TemplateMatcher:
    def __init__(self, templates_directory):
        self.template_images = []
        for image_name in os.listdir(templates_directory):
            self.template_images.append(cv2.imread(os.path.join(templates_directory, image_name), 0))

    
    def check_matching(self, image, template, rot=360, threshold = 0.8):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for ang in range(0, 360, rot):
            rot_template = imutils.rotate_bound(template, ang)

            w, h = rot_template.shape[::-1]

            res = cv2.matchTemplate(img_gray, rot_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            
            # cv2.imshow("image", image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            if len(list(zip(*loc[::-1]))) != 0:
                return True
        return False


    def classificate(self, image_path):
        image = cv2.imread(image_path)
        for template in self.template_images:
            if self.check_matching(image, template):
                return True
        return False 



class NTIDetector:
    def __init__(self):
        self.classifier = SVMHOGClassifier()
        self.detector = RCNNDetector()


    def classificate(self, image_path):
        image = detecto_utils.read_image(image_path)

        labels, boxes, scores = self.detector.predict(image)
        if len(boxes) == 0:
            return False

        if DEBUG:
            detecto_visualize.show_labeled_image(image, boxes, labels)

        labels_numpy = boxes.detach().cpu().numpy()

        for box in boxes:
            detected_logo = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            if self.classifier.classificate(detected_logo):
                return True

        return False    

    
    def classificate_from_url(self, url):
        image_path = os.path.join(WORKING_DIRECTORY, "pred_image.img")
        urllib.request.urlretrieve(url, image_path)
        prediction = self.classificate(image_path)
        os.remove(image_path)
        return prediction
