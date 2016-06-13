#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os
import sys
import cv2
import argparse
from os import listdir
from os.path import isfile, join


directory = None


class ReadableDirectory(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def process_images(file):
    has_faces = False
    try:
        # load in the classifier for face detection
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier('/home/jonathan/Documents/cloudguard/external/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        # read in the image
        img = cv2.imread(file)
        # make a grayscale version of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect the faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.13,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        # display message if found
        for face in faces:
            has_faces = True
        if has_faces:
            print('detection at \'{}\''.format(file))
    except:
        print('Unexpected error processing \'{}\':'.format(file, sys.exc_info()[0]))

    if has_faces:
        return (1, file)
    else:
        return (0, file)


def main(args):
    """Main wrapper function.
    """
    from multiprocessing import Pool
    import shutil

    pool = Pool(processes=4)

    storage_path = os.path.abspath(args.output)
    if storage_path == os.getcwd():
        storage_path = join(storage_path, 'humfilt-results')
    # normalize the path for saving the output
    storage_path = os.path.abspath(storage_path)

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    files = [join(args.directory, f) for f in listdir(args.directory) if isfile(join(args.directory, f))]
    num_detection_files = 0

    for detected, file in pool.imap_unordered(process_images, files):
        num_detection_files += detected
        if detected:
            shutil.copy2(file, join(file, storage_path))

    print('done\n{}/{} images with face detections'.format(num_detection_files, len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine if an image contains a person\'s face')
    parser.add_argument('directory', action=ReadableDirectory, help='Directory where images are located')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Directory where detected images are copied into')
    args = parser.parse_args()
    main(args)
