# NOTE: This script, for some reason, will occasionally fail to run with error
# "malloc(): mismatching next->prev_size (unsorted)". I believe this is due to
# something in the AprilTag library.
# Rerunning the script will eventually work.

from utils.april import check_images, calibrate
import glob

APRILBOARD_PATH = 'data/AprilBoards.pkl'

def main():
    print("[WARNING] This script will occasionally fail with error \"malloc(): mismatching next->prev_size (unsorted)\", usually the first time it is run. I believe this is due to something internal within the AprilTag library. Rerunning the script will eventually work.")
    images = glob.glob('./data/real/aprilboards/img*.jpg')
    valid_files = check_images(images, 'coarse', APRILBOARD_PATH)
    matrix = calibrate(valid_files, 'coarse', APRILBOARD_PATH)
    print(matrix)

if __name__ == "__main__":
    main()
