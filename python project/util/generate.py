import cv2
import numpy as np

from io import BytesIO
from PIL import Image


def PIL_image2image_file(img):
    """
    transform image type to train model
    (모델에 들어가기 위한 이미지 형 변환)

    Args:
         img : PIL.Image.Image type

    Returns:
        PIL.JpegImagePlugin.JpegImageFile type
    """
    bi = BytesIO()
    img.save(bi, format="jpeg")
    return Image.open(bi)


def numpy_arr2PIL_image(arr):
    """
    transform numpy to PIL image
    (numpy화된 이미지를 PIL.Image.Image타입으로 변경)

    Args:
        arr : numpy format image

    Returns:
        PIL.Image.Image type
    """
    return Image.fromarray(arr)


def rand_bbox(size, lam):
    """
    divide area
    (영역 분할)

    Args:
        size : image size

    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_up(img1_path, img2_path, alpha=0.5):
    """
    image mix up

    Args:
        img1_path : mixed by alpha ratio
        img2_path : mixed by 1 - alpha ratio
        alpha : img1 ratio

    Returns:
        img : PIL.JpegImagePlugin.JpegImageFile type
        alpha : image1 ratio
        beta : image2 ratio
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    beta = 1.0 - alpha

    dst = cv2.addWeighted(img1, alpha, img2, beta, 0)
    img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    img = numpy_arr2PIL_image(img)
    img = PIL_image2image_file(img)
    return img, alpha, beta


def cut_mix(image_batch, beta = 1.0):
    """
    image cut mix

    Args:
        image_batch : candidate cut_mix images
        beta  : generate random number

    Returns:
        img_list : updated image
        lam : image1 ratio
    """
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch)) # cut mix 50% probability if image_batch 2
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    image_batch_updated = image_batch.copy()
    image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]

    # calculate lam by pixel
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))

    img_list = []
    for i in range(len(image_batch_updated)):
        img_list.append(PIL_image2image_file(numpy_arr2PIL_image(image_batch_updated[i])))

    return img_list, lam
