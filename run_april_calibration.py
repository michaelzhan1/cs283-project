from utils.april import check_images, calibrate
import glob

APRILBOARD_PATH = 'data/AprilBoards.pkl'

def main():
    images = glob.glob('./data/real/aprilboards/old/img*.jpg')
    valid_files = check_images(images, 'coarse', APRILBOARD_PATH)
    print(valid_files)
    matrix = calibrate(valid_files, 'coarse', APRILBOARD_PATH)
    print(matrix)

if __name__ == "__main__":
    main()
