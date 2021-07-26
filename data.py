import os
import cv2
import torch

import numpy as np
import albumentations as albu
import skimage.io as sio
import matplotlib.pyplot as plt

from skimage.transform import resize
from torch.utils.data import Dataset as BaseDataset


# Target size
target_size = [256, 256]


# Data loader
class Dataset(BaseDataset):
    """
    CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    CLASSES = ['0', '1']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)  # anon012_16205.png
        self.ids.sort()
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '_mask.png') for image_id in
                          self.ids]
        # print('self.masks_fps', self.masks_fps[0])  # ./FP/trainannot/anon012_16205.png

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """ read data
        2 versions use different modules:
            [1] opencv-python
            [2] skimage
        """
        # Version 1: cv2 (Package Name: opencv-python)
        # NOTE: Version 1 often has errors due to opencv verion issues.
        # When the file name of the image is misspelled and the file does not exist,
        # the program code will get stuck and does not report an error.
        # So we use verison 2 using skimage to read data in this project.

        # image = cv2.imread(self.images_fps[i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
        #
        # mask = cv2.imread(self.masks_fps[i], 0) # 0 will return the gray image, e.g. single channel image
        # mask = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)

        # plt.figure()
        # plt.imshow(mask)
        # plt.show()

        # xinter = (np.amax(mask) + np.amin(mask)) / 2

        # masks = [~(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1)
        # print(self.class_values)

        # xinter = 16.5
        # mask = (mask > xinter).astype('float')
        # mask = mask[:, :, np.newaxis]

        # Version 2: skimage (Package Name: scikit-image)
        image = sio.imread(self.images_fps[i])      # [target_size[0], target_size[1], 3]
        image = resize(image, (target_size[0], target_size[1]))

        mask = sio.imread(self.masks_fps[i], as_gray=True)  # read grayscale masks.
        # Binarize mask using grayscale median, which is 0.05 for masks in this project.
        # xinter = (np.amax(mask) + np.amin(mask)) / 2
        # print(xinter)
        xinter = 0.05
        mask = (mask > xinter).astype('float')
        mask = resize(mask, (target_size[0], target_size[1]))

        # expand [target_size[0], target_size[1]] to [target_size[0], target_size[1], 1]
        mask = mask[:, :,  np.newaxis]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    """ make data augmentation """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], always_apply=True, border_mode=0),
        albu.RandomCrop(height=target_size[0], width=target_size[1], always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),  # original
        # albu.augmentations.transforms.GaussNoise(p=0.2),
        albu.IAAPerspective(p=0.5),  # original
        # albu.augmentations.geometric.transforms.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),  # original
                # albu.augmentations.transforms.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(target_size[0], target_size[0])
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.title(' '.join(name.split('_')).title())
        plt.axis('off')     # axis off
        if name == 'image':
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')

    plt.show()


if __name__ == '__main__':
    # sample visualize in train dataset
    x_train_dir = 'data/train/'
    y_train_dir = 'data/trainannot/'
    dataset1 = Dataset(images_dir=x_train_dir, masks_dir=y_train_dir, classes=['0'])
    image, mask = dataset1[0]
    visualize(
        image=image,
        mask=mask.squeeze(),
    )
