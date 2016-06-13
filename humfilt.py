#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2


def main(args):
    """Main wrapper function.
    """
    # load in the classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # loop of each image filename that was given
    for image in args.image:
        # read in the image
        img = cv2.imread(image.name)
        # make a grayscale version of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect the faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        print(type(faces))
        print(faces)
        if faces:
            print('faces found!')
            # show the image
            cv2.imshow(image.name + ' steps', cv2.resize(img, (0,0), fx=0.65, fy=0.65))
            print('Displaying \'{}\'. Press any key to exit.'.format(image.name))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Determine if an image contains a person\'s face')
    parser.add_argument('image', type=argparse.FileType('r'), help='Image to operate on')
    args = parser.parse_args()
    main(args)
