import glob
from utils.calibration import calibrate_from_real_images

def main():
    # images = glob.glob('./data/real/scene/scene*.jpg')
    images = [f'./data/real/scene/img{i}.jpg' for i in range(14)]
    images += [f'./data/real/scene/img_extra{i}.jpg' for i in range(4)]
    K = calibrate_from_real_images(images)
    print(K)


if __name__ == "__main__":
    main()