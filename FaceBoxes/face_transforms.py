######################################################################
# Additional utility code.
# Company: Oxolo GmBH
# Developers: Munshi Harsh, Zaveri Chintan
#####################################################################


"""
A file for all the video based face cropping and saving functionality
This includes:
    - Face detection and cropping
    - Image loading and saving
    - saving frames directly as a video file
"""

from .FaceBoxes import FaceBoxes
import cv2
import torch
import sys, os
from os.path import join

class FaceCrop(FaceBoxes):
    def __init__(self, save_path):
        super().__init__(FaceCrop)
        self.save_path = join(os.getcwd, save_path)
        self.write_size = None
        self.image_buffer = []

    def detect(self, image):
        """
        Detects the face using faceboxes given in the open source
        Args:
            image: (np.ndarray) image in which face needs to be detected
        
        Returns:
            (xmin, ymin, xmax, ymax): (int, int, int, int) the bounding box parameters
        """

        dets = FaceBoxes(image)
        return dets
    
    def crop_image(self, image, step, dets):
        """
        Crops the image #step sizes more than the detected bounding box
        Args:
            image: (np.ndarray) image to be cropped
            step: (int) the number of pixels more than the detected value to crop
            dets: (int, int, int, int) detections from the detect function

        Returns:
            image: (np.ndarray) cropped image
        """

        # assming there is only one face in an image
        xmin, ymin, w, h = dets[0]
        return image[ymin:ymin+h+step, xmin:xmin+w+step]
    
    def resize(self, image, size):
        """
        Resizes the image to the given size
        Args:
            image: (np.ndarray) cropped image that needs to be resized
            size: (int, int) width and height to which it needs to be resized
        
        Returns:
            None
        """
        self.size = size
        self.image_buffer.append(cv2.resize(image, (size[0], size[1])))

    def writeVideo(self):
        """
        writes the image buffer as a video file
        Args:
            None
        
        Returns:
            None
        """
        output_path = join(self.save_path, 'output.avi')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.image_buffer)):
            out.write(self.image_buffer[i])

        out.release()
    
def unit_test(videofile, save_path):
    # Make the videocapture element
    cap = cv2.VideoCapture(videofile)

    cropObject = FaceCrop(save_path)
    while(cap.isOpened()):
        _, frame = cap.read()
        dets = cropObject.detect(frame)
        cropped_image = cropObject.crop_image(frame, 30, dets)
        cropObject.resize(cropped_image, (300,300))
    
    # write the video
    cropObject.writeVideo()

if __name__=="__main__":
    save_path = "cropped_vids"
    videofile = sys.argv[1]
    unit_test(videofile, save_path)