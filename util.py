import os

import numpy as np
from numpy.ma import array
from scipy.misc import imresize

from skimage import data

from Scaler import Scaler


class Util:

    def __init__(self):
        pass

    @staticmethod
    def __load_data_from_dirs_resize(dirs, ext, size):
        files = []
        file_names = []
        count = 0
        for d in dirs:
            for f in os.listdir(d):
                if f.endswith(ext):
                    image = data.imread(os.path.join(d, f))
                    if len(image.shape) > 2:
                        resized_image = imresize(image, size, 'bicubic')
                        files.append(resized_image)
                        file_names.append(os.path.join(d, f))
                    count = count + 1
        return files

    @staticmethod
    def __load_data_from_dirs(dirs, ext):
        files = []
        file_names = []
        count = 0
        for d in dirs:
            for f in os.listdir(d):
                if f.endswith(ext):
                    image = data.imread(os.path.join(d, f))
                    if len(image.shape) > 2:
                        files.append(image)
                        file_names.append(os.path.join(d, f))
                    count = count + 1
        return files

    @staticmethod
    def __load_path(path):
        directories = []
        if os.path.isdir(path):
            directories.append(path)
        for elem in os.listdir(path):
            if os.path.isdir(os.path.join(path, elem)):
                directories = directories + Util.__load_path(os.path.join(path, elem))
                directories.append(os.path.join(path, elem))
        return directories

    @staticmethod
    def __load_data(directory, ext):
        files = Util.__load_data_from_dirs_resize(Util.__load_path(directory), ext, [Scaler.hr_images_dimensions[0], Scaler.hr_images_dimensions[1]])
        return files

    @staticmethod
    def fetch_then_resize_high_resolution_images():
        print("[START] Loading resized HR images...")
        hr_resized_images = Util.__load_data("/content/drive/My Drive/Informatique/Projets_Informatiques/Projets_Python/srgan/input_original_hr_images", ".jpg")
        print("[END] Loading resized HR images...")
        return hr_resized_images

    @staticmethod
    def fetch_training_set_resized_high_resolution_images(resized_hr_images):
        return array(resized_hr_images)

    @staticmethod
    def fetch_testing_set_resized_high_resolution_images(resized_hr_images):
        return array(resized_hr_images)

    @staticmethod
    def fetch_training_set_resized_low_resolution_images(resized_hr_images):
        images = []
        for img in range(len(resized_hr_images)):
            images.append(imresize(resized_hr_images[img], [Scaler.hr_images_dimensions[0] // 4, Scaler.hr_images_dimensions[1] // 4], interp='bicubic', mode=None))
        images_lr = array(images)
        return images_lr

    @staticmethod
    def fetch_testing_set_resized_low_resolution_images(resized_hr_images):
        images = []
        for img in range(len(resized_hr_images)):
            images.append(imresize(resized_hr_images[img], [Scaler.hr_images_dimensions[0] // 4, Scaler.hr_images_dimensions[1] // 4], interp='bicubic', mode=None))
        images_lr = array(images)
        return images_lr

    @staticmethod
    def denormalize(input_data):
        input_data = (input_data + 1) * 127.5
        return input_data.astype(np.uint8)

    @staticmethod
    def normalize(image):
        normalized_image = (image.astype(np.float32) - 127.5) / 127.5
        return normalized_image

    @staticmethod
    def get_random_batch_images_hr_and_lr(batch_size, x_train_hr, x_train_lr):
        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        batch_image_hr = x_train_hr[rand_nums]
        batch_image_lr = x_train_lr[rand_nums]
        return batch_image_hr, batch_image_lr
