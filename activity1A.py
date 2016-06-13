#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np


def rotate(img, angle=180):
    """Simple function that returns a given image rotated by a given angle.
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by the given amount and return it
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def main(args):
    """Main wrapper function.
    """
    # load in the classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # loop of each image filename that was given
    for image in args.image:
        # read in the image
        img = cv2.imread(image.name)
        # make some copies of the image for saving each step
        img_box = img.copy(); img_blur = img.copy()
        img_face_rotate = img.copy()
        img_face_scale1 = img.copy(); img_face_scale2 = img.copy()
        # make a grayscale version of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # set the image's width and height
        (hh, ww) = gray.shape
        text_position = (int(ww/2.5), int(9.75*(hh/10)))
        # detect the faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            y -= int(0.4 * h); h = int(1.8 * h)
            x -= int(0.3 * w); w = int(1.8 * w)
            # select the region of interest
            roi = img[y:y+h, x:x+w]
            # draw the bounding box
            cv2.rectangle(img_box, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img_box, '1', text_position, font, 2, (255,255,255), 2)
            # blur the roi
            img_blur[y:y+h, x:x+w] = cv2.blur(roi, (14,14))
            cv2.putText(img_blur, '2', text_position, font, 2, (255,255,255), 2)
            img_face_rotate[y:y+h, x:x+w] = rotate(roi, 180)
            cv2.putText(img_face_rotate, '3b', text_position, font, 2, (255,255,255), 2)
            # scale down the face
            scale_factor = 0.75 # a scale_factor of 1 would be the original scale
            img_scaled = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
            (rows, cols) = img_scaled.shape[:2]
            r_s = int((hh - rows) / 2); c_s = int((ww - cols) / 2)
            # black out the areas of the image that will be zero padded
            img_face_scale1[0:r_s] = 0; img_face_scale1[:,0:c_s] = 0
            img_face_scale1[r_s+rows:] = 0; img_face_scale1[:,c_s+cols:] = 0
            # assign the scaled down image
            img_face_scale1[r_s:r_s+rows, c_s:c_s+cols] = img_scaled
            cv2.putText(img_face_scale1, '4a', text_position, font, 2, (255,255,255), 2)
            # resize the face, but don't zero pad anything
            h1 = int(h * scale_factor); pad_h = int((h - h1) / 2);
            w1 = int(w * scale_factor); pad_w = int((w - w1) / 2);
            h2 = int(h + (2 * pad_h)); w2 = int(w + (2 * pad_w))
            y2 = y - pad_h; x2 = x - pad_w
            img_face_scale2[y:y+h, x:x+w] = cv2.resize(img[y2:y2+h2, x2:x2+w2], roi.shape[:2])
            cv2.putText(img_face_scale2, '4b', text_position, font, 2, (255,255,255), 2)
        # rotate the original image
        img_rotate = rotate(img, 180)
        cv2.putText(img_rotate, '3a', text_position, font, 2, (255,255,255), 2)
        # show all of the steps beside each other
        vis1 = np.concatenate((img_box, img_blur, img_rotate), axis=1)
        vis2 = np.concatenate((img_face_rotate, img_face_scale1, img_face_scale2), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
        # show a scaled down version of the concatenated images so it fits on the screen
        cv2.imshow(image.name + ' steps', cv2.resize(vis, (0,0), fx=0.65, fy=0.65))
        print('Displaying \'{}\'. Press any key to exit.'.format(image.name))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Activity 1 for Computational Photography by Jonathan Jones')
    parser.add_argument('image', nargs='+', type=argparse.FileType('r'), help='Image to operate on')
    args = parser.parse_args()
    main(args)
