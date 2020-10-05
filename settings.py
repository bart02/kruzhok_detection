import os
from google_drive import download_file


DEBUG = False


UPDATE_LIST_DRIVE_ID = "1MSNgvLpEmXtdWjtfR2FIZzJTFSZEgSPO"

download_file(UPDATE_LIST_DRIVE_ID, "update_list.txt")
weights_ids = open("update_list.txt").read().split()


WORKING_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
MODELS_PATH = os.path.join(WORKING_DIRECTORY, "models")
TEMPLATES_PATH = os.path.join(WORKING_DIRECTORY, "templates")


SVM_MODEL_DRIVE_ID = weights_ids[0] #"18AidWTt_NH-oue3_ZGIGPkwfU5THlLt-"
SVM_MODEL_FILE_PATH = os.path.join(MODELS_PATH, "svm_classifier.joblib")
# RCNN_MODEL_DRIVE_ID = "1tKcwTnL50WRWnI4lrHem8j685tSfepzh"
RCNN_MODEL_DRIVE_ID = weights_ids[1] #"1PavJSFgNX7UeaLnw_jMAAPDpJpIdU-en"
RCNN_MODEL_FILE_PATH = os.path.join(MODELS_PATH, "rcnn.pth")