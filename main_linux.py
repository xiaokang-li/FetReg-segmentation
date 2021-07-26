import os
import random
import torch
import argparse

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

    # add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help='Select "train", "test"or "predict" ')
    parser.add_argument('--model_type', type=str, default="Unet",
                        help='Select model "Unet", "UnetPlusPlus", or "DeepLabV3"')
    parser.add_argument('--encoder', type=str, default="vgg16",
                        help='Select encoder "vgg16", "resnet50", or "resnet101"')
    parser.add_argument('--loss_type', type=str, default='dice',
                        help='Loss function, "dice", "jaccard", "bce" or "bce_logits"')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate in Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of images in each batch')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='Path for models to save, such as "models/"')
    parser.add_argument('--test_image_index', type=int, default=0,
                        help='index for test image in test dir, such as 0')
    parser.add_argument('--GPU_index', type=str,
                        help='Cuda visible devices, index of GPU used, such as "0", "0, 1, 2"')

    # load args
    args = parser.parse_args()
    mode = args.mode
    model_type = args.model_type
    encoder = args.encoder
    loss_type = args.loss_type
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_path = args.model_path
    test_image_index = args.test_image_index
    GPU_index = args.GPU_index

    # initial random seed
    init_seeds(seed=1234)

    # Gpu device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

    if mode == 'train':
        training(x_train_dir=x_train_dir, y_train_dir=y_train_dir,
                 x_valid_dir=x_valid_dir, y_valid_dir=y_valid_dir,
                 MODEL_TYPE=model_type, ENCODER=encoder,
                 ENCODER_WEIGHTS='imagenet',
                 LOSS=loss_type, LEARNING_RATE=lr,
                 EPOCHS=num_epochs, train_batch_size=batch_size,
                 model_save_path=model_path,
                 )

    elif mode == 'test':
        get_test_metrics(x_test_dir=x_test_dir, y_test_dir=y_test_dir,
                         MODEL_TYPE=model_type,
                         ENCODER=encoder,
                         ENCODER_WEIGHTS='imagenet',
                         CLASSES=['0'],
                         LOSS=loss_type,
                         LEARNING_RATE=lr,
                         model_save_path=model_path,
                         )

    elif mode == 'predict':
        demo_predict(x_test_dir=x_test_dir,
                     y_test_dir=y_test_dir,
                     test_image_index=test_image_index,
                     MODEL_TYPE=model_type,
                     ENCODER=encoder,
                     ENCODER_WEIGHTS='imagenet',
                     CLASSES=['0'],
                     LOSS=loss_type,
                     LEARNING_RATE=lr,
                     model_save_path=model_path,
                     )
    else:
        print('mode error, should be "train", or "predict" or "test"')


