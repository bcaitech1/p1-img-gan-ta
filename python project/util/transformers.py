from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.56019358, 0.52410121, 0.501457),
                   std=(0.23318603, 0.24300033, 0.24567522), transform_type='basic'):
    """
    data Augmentation trainsformer

    Args:
        need: kinds of transformer - train, validation
        img_size: Augmentation image size
        mean: mean RGB
        std: Standard Deviation RGB
        transform_type : train transformer kind

    Returns:
        transformations: transformer dictionary (train, val)
    """
    transformations = {}
    if ('train' in need) and (transform_type == 'basic'):
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            CenterCrop(350, 350),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    elif ('train' in need) and (transform_type == 'clahe'):
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            CenterCrop(350, 350),
            CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    elif ('train' in need) and (transform_type == 'allclahe'):
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            CenterCrop(350, 350),
            CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), always_apply=False, p=1.0),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    elif ('train' in need) and (transform_type == 'hor_flip'):
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            CenterCrop(350, 350),
            HorizontalFlip(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    elif ('train' in need) and (transform_type == 'both'):
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            CenterCrop(350, 350),
            CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


    if 'val' in need:
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            CenterCrop(350, 350),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    return transformations
