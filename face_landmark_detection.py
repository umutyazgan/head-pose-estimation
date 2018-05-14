from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
import sys
import os
import dlib
import glob
import numpy as np
import cv2

#  check argument count before starting program
if len(sys.argv) != 2:
    print("invalid arg count")
    exit()

# set predictor path to path of pre-trained .dat file, supplied as 1st argument
predictor_path = sys.argv[1]

# load dlib's face detector
detector = dlib.get_frontal_face_detector()

# load facial landmark detector given in command line argument
predictor = dlib.shape_predictor(predictor_path)

# initialize a video stream object and wait 2 secs to give starting time to camera
vs = VideoStream().start()
time.sleep(2.0)

# video streaming loop
while True:
    # get a single frame from video
    frame = vs.read()

    # resize video width to 400px
    frame = imutils.resize(frame, width=400)

    # turn frame into grayscale
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get size of the frame(used for determining camera specifications)
    size = gs_frame.shape

    # detect faces as rectangles in frame and store them in faces list
    faces = detector(gs_frame, 1)

    # traverse faces in list
    for face in faces:
        # detect facial landmarks in the face rectangles using the
        # predictor and return a shape object containing landmarks as
        # 'parts'
        shape = predictor(gs_frame, face)
        #shape = face_utils.shape_to_np(shape)

        # 2D image points stored in a numpy array. Each 'shape.part(n)'
        # corresponds to a landmark and '.x' and '.y' corresponds to
        # cartesien coordinates of the landmark
        image_points = np.array([
                                    (shape.part(30).x,shape.part(30).y),     # Nose tip
                                    (shape.part(8).x,shape.part(8).y),     # Chin
                                    (shape.part(36).x,shape.part(36).y),     # Left eye left corner
                                    (shape.part(45).x,shape.part(45).y),     # Right eye right corne
                                    (shape.part(48).x,shape.part(48).y),     # Left Mouth corner
                                    (shape.part(54).x,shape.part(54).y)      # Right mouth corner
                                ], dtype="double")
         
        # Predetermined 3D model points for a generic face, taking nose
        # as the origin point.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])
         
        # Approximate camera specifications. focal_length is the image
        # length, center is the center point of image
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
         
        # solve perspective-n-point problem with given inputs(need to
        # check the math behind it)
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
         
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         
        # draw the pose line on frame
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
        # I'm not sure but this might be doing the exact same thing as
        # above 2 lines. 
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255,0,0), 2)

    # draw the frame to the screen
    cv2.imshow("Frame", frame)

    # quit the video stream when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close all windows and stop video stream
cv2.destroyAllWindows()
vs.stop()
