import torch

import numpy as np
import skimage.io as sio
import segmentation_models_pytorch as smp

from skimage.transform import resize
from torch.utils.data import DataLoader

from data import Dataset, visualize, get_preprocessing


""""
Evaluate Metircs:
    Dice = 2 * TP / ((Tp + FN) + (TP + FP)
    IoU = TP / (TP + FN + FP) = Dice / (2 - Dice)
    Sensitivity = TP / (TP + FN)
    PPV = TP / (TP + FP)
"""


def compute_dice(pred, label):
    """ Dice Coefficient"""
    smooth = 1e-5
    m1 = pred.flatten()  # Flatten
    m2 = label.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def compute_IoU(pred, label):
    """ IoU, also known as Jaccard Index """
    smooth = 1e-5
    m1 = pred.flatten()  # Flatten
    m2 = label.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    U = m1.sum() + m2.sum() - intersection
    return (intersection + smooth) / (U + smooth)


def compute_Sensitivity(pred, label):
    """ Sensitivity = TP / (TP + FN) """
    smooth = 1e-5
    m1 = pred.flatten()  # Flatten
    m2 = label.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (intersection + smooth) / (m2.sum() + smooth)


def compute_PPV(pred, label):
    """ Positive predictive value (PPV) = TP / (TP + FP) """
    smooth = 1e-5
    m1 = pred.flatten()  # Flatten
    m2 = label.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (intersection + smooth) / (m1.sum() + smooth)


def training(x_train_dir,
             y_train_dir,
             x_valid_dir,
             y_valid_dir,
             MODEL_TYPE='Unet',
             ENCODER='vgg16',
             ENCODER_WEIGHTS='imagenet',
             ACTIVATION='sigmoid',
             CLASSES=['0'],
             LOSS='dice',
             LEARNING_RATE=1e-4,
             EPOCHS=40,
             train_batch_size=8,
             model_save_path='models/'
             ):
    # model save name
    model_save_name = model_save_path + 'best_model_' + \
                      MODEL_TYPE + '_' + \
                      ENCODER + '_' + \
                      LOSS + '_lr_' + \
                      str(LEARNING_RATE) + '.pth'

    # print information
    print('####')
    print('Training: ')
    print('Model will be saved at:  %s' % model_save_name)
    print()

    DEVICE = ''
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print('Device error, cuda not found.')
        return -1

    # define model
    if MODEL_TYPE == 'Unet':
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif MODEL_TYPE == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif MODEL_TYPE == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    else:
        print("model type error, should be in ['unet', 'UnetPlusPlus', 'DeepLabV3']")
        return -1

    # preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # train data
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # valid data
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # loss
    if LOSS == 'dice':
        loss = smp.utils.losses.DiceLoss()
    elif LOSS == 'jaccard':
        loss = smp.utils.losses.JaccardLoss()
    elif LOSS == 'bce':
        loss = smp.utils.losses.BCELoss()
    elif LOSS == 'bce_logits':
        loss = smp.utils.losses.BCEWithLogitsLoss()
    else:
        print("loss error, should be in ['dice', 'jaccard', 'bce', 'bce_logits']")
        return -1

    # metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model
    # save at the best valid score
    max_score = 0
    save_epoch = 0

    for i in range(EPOCHS):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save at the best
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, model_save_name)
            save_epoch = i
            print()
            print('#######')
            print('Epoch %2d, Model saved at %s' % (i, model_save_name))
            print('#######')

        if i == int(0.6 * EPOCHS):
            optimizer.param_groups[0]['lr'] = LEARNING_RATE / 10
            print('Decrease decoder learning rate to %f!' % (LEARNING_RATE / 10))

    # print save information
    print()
    print('Final model saved at: %s' % model_save_name)
    print('Saved at Epoch %d' % save_epoch)


def get_test_metrics(x_test_dir,
                     y_test_dir,
                     MODEL_TYPE='Unet',
                     ENCODER='vgg16',
                     ENCODER_WEIGHTS='imagenet',
                     CLASSES=['0'],
                     LOSS='dice',
                     LEARNING_RATE=1e-4,
                     model_save_path='models/'
                     ):
    """
    Predict for test dataset and compute evaluate metrics.
    :param x_test_dir:
    :param y_test_dir:
    :param MODEL_TYPE:
    :param ENCODER:
    :param ENCODER_WEIGHTS:
    :param CLASSES:
    :param LOSS:
    :param LEARNING_RATE:
    :return:
    """
    # model name
    model_save_name = model_save_path + 'best_model_' + \
                      MODEL_TYPE + '_' + \
                      ENCODER + '_' + \
                      LOSS + '_lr_' + \
                      str(LEARNING_RATE) + '.pth'

    # print information
    print()
    print('Test metrics on model: %s' % model_save_name)

    DEVICE = ''
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print('Device error, cuda not found.')
        return -1

    # preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # model prepare
    best_model = torch.load(model_save_name)

    # data prepare
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        # augmentation=get_validation_augmentation(),
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # masks fps
    masks_fps = test_dataset.masks_fps

    # define metrics
    length = len(masks_fps)
    dices = np.zeros((length,))
    ious = np.zeros((length,))
    sens = np.zeros((length,))
    ppvs = np.zeros((length,))

    # loop for test
    original_size = [470, 470]  # original size of masks
    for i in range(length):
        mask_ori = sio.imread(masks_fps[i], as_gray=True)
        mask = (mask_ori > 0.05).astype('float')
        mask = resize(mask, (original_size[0], original_size[1]))

        # predict
        image, _ = test_dataset[i]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pred = resize(pr_mask, (original_size[0], original_size[1]))
        pred[pred >= 0.5] = 1.
        pred[pred < 0.5] = 0.

        # compute metrics
        dice = compute_dice(pred=pred, label=mask)
        iou = compute_IoU(pred=pred, label=mask)
        sen = compute_Sensitivity(pred=pred, label=mask)
        ppv = compute_PPV(pred=pred, label=mask)

        # save metrics
        dices[i] = dice
        ious[i] = iou
        sens[i] = sen
        ppvs[i] = ppv

    print('#######')
    print('model = %s' % model_save_name)
    print('encoder = %s' % ENCODER)
    print('dice = %.2f ± %.2f' % (np.mean(dices) * 100, np.std(dices) * 100))
    print('iou = %.2f ± %.2f' % (np.mean(ious) * 100, np.std(ious) * 100))
    print('sen = %.2f ± %.2f' % (np.mean(sens) * 100, np.std(sens) * 100))
    print('ppv = %.2f ± %.2f' % (np.mean(ppvs) * 100, np.std(ppvs) * 100))



def demo_predict(x_test_dir,
                 y_test_dir,
                 test_image_index=0,
                 MODEL_TYPE='Unet',
                 ENCODER='vgg16',
                 ENCODER_WEIGHTS='imagenet',
                 CLASSES=['0'],
                 LOSS='dice',
                 LEARNING_RATE=1e-4,
                 model_save_path='models/'
                 ):
    # model name
    model_save_name = model_save_path + 'best_model_' + \
                      MODEL_TYPE + '_' + \
                      ENCODER + '_' + \
                      LOSS + '_lr_' + \
                      str(LEARNING_RATE) + '.pth'

    # print information
    print()
    print('Predict image %d on model: %s' % (test_image_index, model_save_name))
    print()

    DEVICE = ''
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print('Device error, cuda not found.')
        return -1

    # preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # model prepare
    best_model = torch.load(model_save_name)

    # data prepare
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        # augmentation=get_validation_augmentation(),
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # mask
    original_size = [470, 470]  # original size of masks
    images_fps = test_dataset.images_fps
    masks_fps = test_dataset.masks_fps

    image_ori = sio.imread(images_fps[test_image_index])
    image_ori = resize(image_ori, (original_size[0], original_size[1]))
    mask_ori = sio.imread(masks_fps[test_image_index], as_gray=True)
    mask = (mask_ori > 0.05).astype('float')
    mask = resize(mask, (original_size[0], original_size[1]))

    # predict
    image, _ = test_dataset[test_image_index]
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    print('pr_mask shape: ', pr_mask.shape)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    print('pr_mask shape 2: ', pr_mask.shape)
    pred = resize(pr_mask, (original_size[0], original_size[1]))
    pred[pred >= 0.5] = 1.
    pred[pred < 0.5] = 0.

    # compute metrics
    print('pred shape: ', pred.shape)
    print('mask shape: ', mask.shape)
    dice = compute_dice(pred=pred, label=mask)
    iou = compute_IoU(pred=pred, label=mask)
    sen = compute_Sensitivity(pred=pred, label=mask)
    ppv = compute_PPV(pred=pred, label=mask)

    # print
    print('image %s' % images_fps[test_image_index])
    print('Dice = %.2f' % (dice * 100))
    print('IoU = %.2f' % (iou * 100))
    print('Sen = %.2f' % (sen * 100))
    print('PPV = %.2f' % (ppv * 100))

    # visualize
    visualize(image=image_ori,
              mask=mask,
              pred=pred)


if __name__ == '__main__':
    import os
    from main_linux import init_seeds

    # Data path
    x_train_dir = 'data/train/'
    y_train_dir = 'data/trainannot/'
    x_valid_dir = 'data/val/'
    y_valid_dir = 'data/valannot/'
    x_test_dir = 'data/test/'
    y_test_dir = 'data/testannot/'

    GPU_index = '0'
    test_image_index = 0
    model_type = 'Unet'
    encoder = 'vgg16'
    loss_type = 'dice'
    lr = 1e-4
    model_path = 'models/'


    # initial random seed
    init_seeds(seed=1234)

    # Gpu device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

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

