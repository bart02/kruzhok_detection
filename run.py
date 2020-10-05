from settings import *
import warnings
from models import NTIDetector, TemplateMatcher
from google_drive import download_file
import argparse
import os
import sys

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    nti_detector = NTIDetector()

    parser = argparse.ArgumentParser(description='NTI logo detection app')

    parser.add_argument('-d', action="store", type=str, dest="dirpath", default="")
    parser.add_argument('-f', action="store", dest="filepath", type=str, default="")

    args = parser.parse_args()

    if args.dirpath == "" and args.filepath != "":
        if nti_detector.classificate(args.filepath):
            print("kruzhok")
    elif args.dirpath != "" and args.filepath == "":
        for image_name in os.listdir(args.dirpath):
            if nti_detector.classificate(os.path.join(args.dirpath, image_name)):
                print(f"{image_name}: kruzhok")
            else:
                print(f"{image_name}: ne kruzhok")
    else:
        parser.print_help()
        print("\nError: please use -d and -f flags separately")