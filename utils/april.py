import os
import glob
import cv2
import numpy as np                      
import pickle
from pupil_apriltags import Detector

def _detect_aprilboard(img, board, apriltag_detector):
    # Usage:  imgpoints, objpoints, tag_ids = detect_aprilboard(img,board,AT_detector)
    #
    # Input: 
    #   image -- grayscale image
    #   board -- at_coarseboard or at_fineboard (list of dictionaries)
    #   AT_detector -- AprilTag Detector parameters
    #
    # Returns: 
    #   imgpoints -- Nx2 numpy array of (x,y) image coords
    #   objpoints -- Nx3 numpy array of (X,Y,Z=0) board coordinates (in inches)
    #   tag_ids -- Nx1 list of tag IDs
    
    imgpoints=[]
    objpoints=[]
    tagIDs=[]
    
    # detect april tags
    imgtags = apriltag_detector.detect(img, 
                                    estimate_tag_pose=False, 
                                    camera_params=None, 
                                    tag_size=None)

    if len(imgtags):
        # collect image coordinates of tag centers
        # imgpoints = np.vstack([ sub.center for sub in tags ])

        # list of all tag_id's that are in board
        brdtagIDs = [ sub['tag_id'] for sub in board ]

        # list of all detected tag_id's that are in image
        imgtagIDs = [ sub.tag_id for sub in imgtags ]

        # list of all tag_id's that are in both
        tagIDs = list(set(brdtagIDs).intersection(imgtagIDs))
        
        if len(tagIDs):
            # all board list-elements that contain one of the common tag_ids
            objs=list(filter(lambda tagnum: tagnum['tag_id'] in tagIDs, board))
            
            # their centers
            objpoints = np.vstack([ sub['center'] for sub in objs ])
    
            # all image list-elements that contain one of the detected tag_ids
            imgs=list(filter(lambda tagnum: tagnum.tag_id in tagIDs, imgtags))    
            
            # their centers
            imgpoints = np.vstack([ sub.center for sub in imgs ])
        
    return imgpoints, objpoints, tagIDs


def check_images(images, board_type, board_file):
    pickle_data = pickle.load(open(board_file, "rb"))
    at_coarseboard = pickle_data['at_coarseboard']
    at_fineboard = pickle_data['at_fineboard']

    if board_type == 'fine':
        num_points = 80
        board = at_fineboard
    elif board_type == 'coarse':
        num_points = 35
        board = at_coarseboard
    else:
        raise ValueError('Invalid board type')
    
    valid_files = []
    for fname in images:
        orig = cv2.imread(fname)
        if len(orig.shape) == 3:
            img = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        else:
            img = orig
            
        # set up april tag detector (I use default parameters; seems to be OK)
        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

        # detect "coarse" April Board only (would need to run again for other board)
        _, objpoints_coarse, _ = _detect_aprilboard(img, board, at_detector)
        
        if len(objpoints_coarse) == num_points:
            valid_files.append(fname)
    return valid_files

def calibrate(images, board_type, board_file):
    pickle_data = pickle.load(open(board_file, "rb"))
    at_coarseboard = pickle_data['at_coarseboard']
    at_fineboard = pickle_data['at_fineboard']

    if board_type == 'fine':
        BOARD = at_fineboard
    elif board_type == 'coarse':
        BOARD = at_coarseboard
    else:
        raise ValueError('Invalid board type')
    
    calObjPoints = []
    calImgPoints = []

    # set up april tag detector (I use default parameters; seems to be OK)
    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    for count,fname in enumerate(images):
        
        # read image and convert to grayscale if necessary
        orig = cv2.imread(fname)
        if len(orig.shape) == 3:
            img = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        else:
            img = orig

        # detect apriltags and report number of detections
        imgpoints, objpoints, tagIDs = _detect_aprilboard(img,BOARD,at_detector)
        
        # uncomment this line for verbose output
        #print("{} {}: {} imgpts, {} objpts".format(count, fname, len(imgpoints),len(objpoints)))
        
        # append detections if some are found
        if len(imgpoints) and len(objpoints):
                
            # append points detected in all images, (there is only one image now)
            calObjPoints.append(objpoints.astype('float32'))
            calImgPoints.append(imgpoints.astype('float32'))

    # convert to numpy array
    calObjPoints = np.array(calObjPoints)
    calImgPoints = np.array(calImgPoints)
        
    ## calibrate the camera 
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = cv2.calibrateCamera(
        calObjPoints, 
        calImgPoints, 
        img.shape,    # image H,W for initialization of the principal point
        None,         # no initial guess for the remaining entries of calMatrix
        None)         # initial guesses for distortion coefficients are all 0
    
    return calMatrix

def main():
    images = [os.path.relpath(f, start='./') for f in sorted(glob.glob('../aprildata/camera_data/april/*.jpg'))]
    valid_files = check_images(images, 'coarse', '../aprildata/AprilBoards.pkl')
    matrix = calibrate(valid_files, 'coarse', '../aprildata/AprilBoards.pkl')
    print(matrix)


if __name__ == "__main__":
    main()