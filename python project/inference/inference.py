import os
import sys
import argparse

import torch

import pandas as pd
from tqdm import tqdm

test_dir = '/opt/ml/input/data/eval'


class ModelExistError(Exception):
    def __init__(self):
        super().__init__('model not exist')


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')


def get_model(model_name):
    """
    get model object

    Args:
        model_name : model name

    Returns:
        model object
    """
    if model_name == 'resnext':
        return models.MyResNext()
    else:
        raise ModelExistError


@torch.no_grad()
def inference_single(model, loader):
    """
    make submission file
    (제출 파일 생성)
    """
    model.eval()

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.type(torch.FloatTensor).to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')

@torch.no_grad()
def inference_multi(model, loader):
    """
    make submission file
    (제출 파일 생성)
    """
    model.eval()

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.type(torch.FloatTensor).to(device)
            pred = model(images) # mask, gender age

            mask_pred = pred[0].argmax(dim=-1)
            gender_pred = pred[1].argmax(dim=-1)
            age_pred = pred[2].argmax(dim=-1)

            pred = mask_pred * 6 + gender_pred * 3 + age_pred

            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='inference parameter')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='model file path')
    parser.add_argument('-n', '--name', default=None, type=str,
                        help='model name')
    parser.add_argument('-t', '--type', default = 'single', type=str,
                        help='inference type')
    parser.add_argument('-tr', '--transform', default = 'basic', type=str,
                        help='transform type')
    

    args = parser.parse_args()

    if (args.path is None) or (args.name is None):
        raise ParameterError

    sys.path.append('/opt/ml/pstage1')
    from model import models
    from dataloader import mask
    from util import transformers

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model = get_model(args.name)
    model.load_state_dict(torch.load(args.path))
    model.cuda()

    # meta 데이터와 이미지 경로
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    transformers = transformers.get_transforms(
        mean=(0.56019358, 0.52410121, 0.501457),
        std=(0.23318603, 0.24300033, 0.24567522),
        transform_type = args.transform
    )
    
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = mask.TestDataset(image_paths, transformers['val'])

    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False
    )

    if args.type == 'single':
        inference_single(model, loader)
    elif args.type == 'multi':
        inference_multi(model, loader)

