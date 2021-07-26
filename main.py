import os
import random
import torch

import numpy as np

from utils import training, get_test_metrics, demo_predict


def init_seeds(seed=1234):
    """ set random seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Data path
    x_train_dir = 'data/train/'
    y_train_dir = 'data/trainannot/'
    x_valid_dir = 'data/val/'
    y_valid_dir = 'data/valannot/'
    x_test_dir = 'data/test/'
    y_test_dir = 'data/testannot/'

    # args
    mode = 'train'          # train, test, predict
    model_type = 'Unet'     # Unet, UnetPLusPlus, DeepLabV3
    encoder = 'vgg16'       # vgg16, resnet50, resnet101, et al.
    loss_type = 'dice'      # "dice", "jaccard", "bce" or "bce_logits"
    lr = '1e-6'
    num_epochs = 40
    batch_size = 12
    model_path = 'models/'
    test_image_index = 0
    GPU_index = '0'

    if mode == 'train':
        training(x_train_dir=x_train_dir, y_train_dir=y_train_dir,
                 x_valid_dir=x_valid_dir, y_valid_dir=y_valid_dir,
                 MODEL_TYPE=model_type, ENCODER=encoder,
                 ENCODER_WEIGHTS='imagenet',
                 LOSS=loss_type, LEARNING_RATE=lr,
                 EPOCHS=num_epochs, train_batch_size=batch_size,
                 model_save_path=model_path,
                 )

    if mode == 'test':
        get_test_metrics(x_test_dir=x_test_dir, y_test_dir=y_test_dir,
                         MODEL_TYPE=model_type,
                         ENCODER=encoder,
                         ENCODER_WEIGHTS='imagenet',
                         CLASSES=['0'],
                         LOSS=loss_type,
                         LEARNING_RATE=lr,
                         model_save_path=model_path,
                         )
    if mode == 'predict':
        demo_predict(x_test_dir=x_test_dir,
                     y_test_dir=y_test_dir,
                     test_image_index=test_image_index,
                     MODEL_TYPE=model_type,
                     ENCODER=encoder,
                     ENCODER_WEIGHTS='imagenet',
                     CLASSES=['0'],
                     LOSS=loss_type,
                     LEARNING_RATE=loss_type,
                     model_save_path=model_path,
                     )


