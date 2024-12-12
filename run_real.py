import glob
from utils.calibration import calibrate_from_real_images

def main():
    images = glob.glob('./data/real/scene/*.jpg')
    K = calibrate_from_real_images(images)
    print(K)


if __name__ == "__main__":
    main()