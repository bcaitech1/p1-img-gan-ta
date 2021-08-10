import os
import sys
import random

import torch
import torch.utils.data as data
from glob import glob

from PIL import Image
import numpy as np

from albumentations import *
from albumentations.pytorch import ToTensorV2


sys.path.append('/opt/ml/pstage01')
from util import generate


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2


class GenderLabels:
    male = 0
    female = 1


class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2


class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
    }

    def __init__(self, data_dir,img_dir, val_ratio=0.2, upsample = False, extend_sample = False):
        """
        initialize mask dataset
        (마스크 기본 데이터 셋 구축)

        Args:
            data_dir : train root dir (standard + external)
            img_dir: train data dir (standard)
            transform: Augmentation function
            upsample : the presence or absence upsample
            extend_sample : the presence or absence extend sample
        """
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.val_ratio = val_ratio

        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.labels = []
        self.upsample = upsample
        self.extend_sample = extend_sample

        self.label2index = dict()

        if self.img_dir != "":
            self.setup(extend_sample=extend_sample)

    def set_transform(self, transform):
        """
        set transform
        (Data Augmentation설정)
        """
        self.transform = transform
        
    def set_list(self, img_path, mask_label, gender_label, age_label, label):
        """
        add data to Dataset attribute
        (데이터셋 각각의 속성들에 정보 저장)

        Args:
            img_path : image path
            mask_label : only mask label 
            gender_label : only gender label
            age_label : only age label
            label : ground truth
        """
        self.image_paths.append(img_path)
        self.mask_labels.append(mask_label)
        self.gender_labels.append(gender_label)
        self.age_labels.append(age_label)
        self.labels.append(label)

    def setup(self, extend_sample= False):
        """
        store image path , label imformations etc...(initialize setting)
        (이미지에 따른 경로 및 부가 메타정보 저장, 초기 세팅 작업)

        Args:
            add_sample : add external material
        """
        # 현재 있는 추가 외부 자료
        external_data_class = [2,5,8,11,14,17]

        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, mask_label in self._file_names.items():
                for ext in IMG_EXTENSIONS:
                    file_name += ext
                    img_path = os.path.join(self.img_dir, profile,file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)

                    if os.path.exists(img_path):
                        id, gender, race, age = profile.split("_")
                        gender_label = getattr(GenderLabels, gender)  # 속성값을 가져옴
                        age_label = AgeGroup.map_label(age)
                        ground_truth = mask_label * 6 + gender_label * 3 + age_label
                        
                        if self.upsample and (ground_truth == 1):
                            self.set_list(img_path, mask_label, gender_label, age_label, ground_truth)
                        self.set_list(img_path, mask_label, gender_label, age_label, ground_truth)

        # 외부 데이터 추가
        if extend_sample:
            for edc in external_data_class:
                external_img_dir = os.path.join(self.data_dir, str(edc))
                external_profiles = [external_profile for external_profile in os.listdir(external_img_dir) if not external_profile.startswith('.')]
                for external_profile in external_profiles:
                    img_path = os.path.join(external_img_dir, external_profile)
                    
                    edc_temp = edc
                    mask_label = int(edc_temp / 6)
                    edc_temp %= 6
                    gender_label = int(edc_temp / 3)
                    edc_temp %= 3
                    age_label = int(edc_temp)

                    if self.upsample and (edc == 1):
                        self.set_list(img_path,mask_label, gender_label, age_label, edc)
                    self.set_list(img_path,mask_label, gender_label, age_label, edc)

        for i, value in enumerate(self.labels):
            if value not in self.label2index.keys():
                self.label2index[value] = [i]
            else:
                self.label2index[value].append(i)

    def __getitem__(self, index):
        """
        call data located at index
        (데이더 셋의 반환 데이터를 설정)

        Args:
            index: Index value of the data to be loaded

        Returns:
            transformed image, label, path
        """
        # call image
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        # call label
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label

        # image Augmentation
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, multi_class_label, image_path

    def __len__(self):
        return len(self.image_paths)

    def split_dataset(self):
        """
        split train and test dataset
        (훈련 데이터와 테스트 데이터를 분리)
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set

    def get_mixup_img(self, img1_paths,labels, label1, label2, alpha = 0.5):
        """
        get mixxup image
        (데이터 mixup 작업)

        Args:
            img1_paths : standard image paths
            labels : ground truths
            label1 : target label
            label2 : target label
            alpha : image1 ratio
        
        Returns:
            res_img : mixuped image
            res_label1 : image1 labels
            res_label2 :  image2 labels
            alpha : image1 ratio 
            beta :  image2 ratio
        """
        res_img = []
        res_label1 = []
        res_label2 = []
        # alpha = 0.5
        # beta = 0.5
        beta = 1 - alpha

        for i,v in enumerate(labels):
            if v == label1:
                candidate_indexs = self.label2index[label2]

                # 예외 처리
                if len(candidate_indexs) == 0:
                    break

                img2_path = self.image_paths[random.choice(candidate_indexs)]
                image, alpha, beta = generate.mix_up(img1_paths[i], img2_path, alpha)
                image_transform = self.transform(image=np.array(image))['image']

                res_img.append(image_transform.numpy())
                res_label1.append(label1)
                res_label2.append(label2)

        res_img = torch.tensor(res_img)
        res_label1 = torch.tensor(res_label1)
        res_label2 = torch.tensor(res_label2)

        return res_img,res_label1, res_label2, alpha, beta 

    def get_cutmix_img(self, img1_paths,labels, label1, label2, beta = 1.0):
        """
        get cutmix image
        (데이터 cutmix 작업)

        Args:
            img1_paths : standard image paths
            labels : ground truths
            label1 : target label
            label2 : target label
            beta : num to create ratio
        
        Returns:
            res_img1 : cutmixed image(lam : image1, 1- lam : image2)
            res_img2 : cutmixed image(lam - 1 : image1, lam : image2)
            res_label1 : image1 labels
            res_label2 : image2 labels
            lam : ratio
            1-lam : ratio
        """
        res_img1 = []
        res_img2 = []
        res_label1 = []
        res_label2 = []
        lam = 0.5

        for i, v in enumerate(labels):
            if v == label1:
                candidate_indexs = self.label2index[label2]

            # 예외 처리
                if len(candidate_indexs) == 0:
                    break
            
                img1_path = img1_paths[i]
                img2_path = self.image_paths[random.choice(candidate_indexs)]
                img_batch = []
                img_batch.append(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB))
                img_batch.append(cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB))
                # print(img_batch)
                img_batch = np.array(img_batch)
                img_list, lam = generate.cut_mix(img_batch, beta)

                img_transform1 = self.transform(image=np.array(img_list[0]))['image']
                img_transform2 = self.transform(image=np.array(img_list[1]))['image']

                res_img1.append(img_transform1.numpy())
                res_img2.append(img_transform2.numpy())
                res_label1.append(label1)
                res_label2.append(label2)

        res_img1 = torch.tensor(res_img1)
        res_img2 = torch.tensor(res_img2)
        res_label1 = torch.tensor(res_label1)
        res_label2 = torch.tensor(res_label2)

            # image, alpha, beta = generate.mix_up(img1_path[i], img2_path, alpha)
            # image_transform = self.transform(image=np.array(image))['image']
            # res_img.append(image_transform.numpy())
            # res_label1.append(label1)
            # res_label2.append(label2)

        # img1_path = self.image_paths[index]
        # candidate_indexs = self.label2index[label2]
        # img2_path = self.image_paths[candidate_indexs[random.randint(0,len(candidate_index))]]
        # img_batch = []
        # img_batch.append(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB))
        # img_batch.append(cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB))
        # img_list, lam = generate.mix_up(img1_path, img2_path)

        return res_img1, res_img2, res_label1, res_label2, lam, 1-lam


class MultiHeadMaskDataset(MaskBaseDataset):
    """
    dataset for multihead classification
    (multihead classification을 위한 데이터셋)
    """


    def __init__(self,data_dir,img_dir, val_ratio=0.2, upsample = False,extend_sample = False):
        super().__init__(data_dir,img_dir,val_ratio,upsample,extend_sample)
    
    def __getitem__(self, index):
        """
        call data located at index
        (데이더 셋의 반환 데이터를 설정)

        Args:
            index: Index value of the data to be loaded
        """
        # call image
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        # call label
        mask_label = self.mask_labels[index] # ground truth
        gender_label = self.gender_labels[index] # ground truth
        age_label = self.age_labels[index] # ground truth

        # image Augmentation
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, mask_label, gender_label, age_label, image_path



class MaskDatasetFold(data.Dataset):
    """
    dataset for k-fold cross validation
    (k-fold cross validation을 위한 데이터셋) 
    """
    def __init__(self, image_dir, info, transform = None):
        self.image_dir = image_dir
        self.info = info
        self.transform = transform

        self.image_paths = [path for name in info.path.values for path in glob(os.path.join(image_dir, name, '*'))]
        self.image_paths = list(filter(self.is_image_file, self.image_paths))
        self.image_paths = list(filter(self.remove_hidden_file, self.image_paths))
        
        self.labels = [self.convert_label(path, sep=False) for path in self.image_paths]
        
    def get_mask_label(self,image_name):
        if 'incorrect_mask' in image_name:
            return 1
        elif 'normal' in image_name:
            return 2
        elif 'mask' in image_name:
            return 0
        else:
            raise ValueError(f'No class for {image_name}')

    def get_gender_label(self,gender):
        return 0 if gender == 'male' else 1

    def get_age_label(self,age):
        return 0 if int(age) < 30 else 1 if int(age) < 60 else 2
    
    def is_image_file(self,filepath):
        """
        check whether it is an image file
        (이미지 파일인지 여부 판별)
        """
        return any(filepath.endswith(extension) for extension in IMG_EXTENSIONS)

    def remove_hidden_file(self,filepath):
        """
        check metadata file
        (메타 데이터 파일인지 여부 파악)
        """
        filename = filepath.split('/')[-1]
        return False if filename.startswith('._') else True

    def convert_label(self,image_path, sep=False):
        """
        calculate ground truth
        (실제 라벨 값 계산)

        Args :
            sep : 분리하여 라벨을 줄지 합쳐서 줄지 판별
        """
        image_name = image_path.split('/')[-1]
        mask_label = self.get_mask_label(image_name)
        # mask_label = getattr(MaskLabels,image_name)

        profile = image_path.split('/')[-2]
        image_id, gender, race, age = profile.split("_")
        gender_label = self.get_gender_label(gender)
        # gender_label =  getattr(GenderLabels, gender)
        age_label = AgeGroup.map_label(age)

        if sep:
            return mask_label, gender_label, age_label
        else:
            return mask_label * 6 + gender_label * 3 + age_label
        
    def get_img(self,path):
        """
        get image numpy type
        
        Args :
            path : image path
        """
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, ::-1]
        return im_rgb

    def __getitem__(self, idx):
        """
        call data located at index
        (데이더 셋의 반환 데이터를 설정)

        Args:
            index: Index value of the data to be loaded
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self.get_img(image_path)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

    def __len__(self):
        return len(self.image_paths)



class TestDataset(data.Dataset):
    """
    inference를 위한 테스트 데이터 셋
    """
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def set_transform(self, transform):
        """
        transform 함수를 설정
        """
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image_transform = self.transform(image=np.array(image))['image']
        return image_transform

    def __len__(self):
        return len(self.img_paths)

class TestDatasetTTA(TestDataset):
    """
    TTA inference를 위한 테스트 데이터 셋
    """
    def __init__(self, img_paths, transforms=None):
        super().__init__(img_paths, transforms)
    
    def set_transform(self, transforms):
        """
        transform 함수를 설정
        """
        self.transform = transforms
    
    def __getitem__(self,index):
        image = Image.open(self.img_paths[index])

        image_transforms = []

        if self.transform:
            for transform in self.transform:
                image_transforms.append(self.transform(image=np.array(image))['image'])

        return image_transforms


if __name__ == '__main__':
    import sys

    sys.path.append('/opt/ml/pstage01')
    from model import models, loss, metric
    from dataloader import mask
    from util import meter, transformers

    from sklearn.model_selection import StratifiedKFold


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
        elif model_name == 'effi':
            return models.MyEfficentNetb4()
        else:
            pass

    def getDataloader(total_dataset,train_idx,valid_idx,batch_size,num_workers,):
        train_set = torch.utils.data.Subset(total_dataset,indices=train_idx)
        val_set = torch.utils.data.Subset(total_dataset,indices=valid_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )
            
        valid_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

        return train_loader, valid_loader

    dataset = mask.MaskBaseDataset(
        img_dir = ""
    )

    # RGB mean, std
    mean, std = (0.56019358, 0.52410121, 0.501457), (0.23318603, 0.24300033, 0.24567522)

    transform = transformers.get_transforms(mean=mean, std=std, transform_type= "both")

    skf = StratifiedKFold(n_splits=5)

    train_loader, valid_loader = mask.getDataloader(
        total_dataset=dataset,
        train_idx=[],
        valid_idx=[],
        batch_size=16,
        num_workers=4,
    )

    # train_loader.dataset.dataset.transformer = transform['train']
    train_loader.dataset.dataset.set_transform(transform['train'])
    valid_loader.dataset.dataset.set_transform(transform['val'])


    print('Done')



# class MaskBaseDatasetFold(data.Dataset):
#     num_classes = 3 * 2 * 3

#     _file_names = {
#         "mask1": MaskLabels.mask,
#         "mask2": MaskLabels.mask,
#         "mask3": MaskLabels.mask,
#         "mask4": MaskLabels.mask,
#         "mask5": MaskLabels.mask,
#         "incorrect_mask": MaskLabels.incorrect,
#         "normal": MaskLabels.normal
#     }

#     def __init__(self, img_dir, profiles, val_ratio=0.2):
#         """
#         initialize mask dataset

#         Args:
#             img_dir: train root dir
#             transform: Augmentation function
#             profiles: peoples profile
#         """
#         self.img_dir = img_dir
#         self.profiles = profiles
#         self.val_ratio = val_ratio

#         self.image_paths = []
#         self.mask_labels = []
#         self.gender_labels = []
#         self.age_labels = []

#         self.setup()

#     def set_transform(self, transform):
#         """
#         set transform
#         """
#         self.transform = transform

#     def setup(self):
#         """
#         store image path and label
#         """
#         #         profiles = os.listdir(self.img_dir)
#         for profile in self.profiles:
#             profile = os.path.join(self.img_dir, profile)
#             for file_name, label in self._file_names.items():
#                 for ext in IMG_EXTENSIONS:
#                     file_name += ext
#                     img_path = os.path.join(self.img_dir, profile,
#                                             file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
#                     if os.path.exists(img_path):
#                         self.image_paths.append(img_path)
#                         self.mask_labels.append(label)

#                         id, gender, race, age = profile.split("_")
#                         gender_label = getattr(GenderLabels, gender)  # 속성값을 가져옴
#                         age_label = AgeGroup.map_label(age)

#                         self.gender_labels.append(gender_label)
#                         self.age_labels.append(age_label)

#     def __getitem__(self, index):
#         """
#         call data located at index

#         Args:
#             index: Index value of the data to be loaded
#         """
#         # call image
#         image_path = self.image_paths[index]
#         image = Image.open(image_path)

#         # call label
#         mask_label = self.mask_labels[index]
#         gender_label = self.gender_labels[index]
#         age_label = self.age_labels[index]
#         multi_class_label = mask_label * 6 + gender_label * 3 + age_label

#         # image Augmentation
#         image_transform = self.transform(image=np.array(image))['image']
#         return image_transform, multi_class_label

#     def __len__(self):
#         return len(self.image_paths)

#     def split_dataset(self):
#         n_val = int(len(self) * self.val_ratio)
#         n_train = len(self) - n_val
#         train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
#         return train_set, val_set


# class MaskSplitByProfileDataset(MaskBaseDataset):
#     def __init__(self, data_dir, mean=(0.56019358, 0.52410121, 0.501457), std=(0.23318603, 0.24300033, 0.24567522), val_ratio=0.2):
#         self.indices = defaultdict(list)
#         super().__init__(data_dir, mean, std, val_ratio)

#     @staticmethod
#     def _split_profile(profiles, val_ratio):
#         """
#         get train and validation index
#         """
#         length = len(profiles)
#         n_val = int(length * val_ratio)

#         val_indices = set(random.choices(range(length), k=n_val))
#         train_indices = set(range(length)) - val_indices
#         return {
#             "train": train_indices,
#             "val": val_indices
#         }

#     def setup(self):
#         profiles = os.listdir(self.data_dir)
#         profiles = [profiles for profile in profiles if not profile.startswith('.')]
#         split_profiles = self._split_profile(profiles, self.val_ratio)

#         cnt = 0

#         for phase, indices in split_profiles.items():
#             for _idx in indices:
#                 profile = profiles[_idx] # 한 사람의 프로파일을 가져옴
#                 img_folder = os.path.join(self.data_dir, profile) # 경로 가져오기
#                 for file_name in os.listdir(img_folder):
#                     _file_name, ext = os.path.splitext(file_name)
#                     if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
#                         continue

#                     img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
#                     mask_label = self._file_names[_file_name]

#                     id, gender, race, age = profile.split("_")
#                     gender_label = getattr(self.GenderLabels, gender)
#                     age_label = self.AgeGroup.map_label(age)

#                     self.image_paths.append(img_path)
#                     self.mask_labels.append(mask_label)
#                     self.gender_labels.append(gender_label)
#                     self.age_labels.append(age_label)

#                     self.indices[phase].append(cnt)
#                     cnt += 1

#     def split_dataset(self):
#         return [Subset(self, indices) for phase , indices in self.indices.items()]
