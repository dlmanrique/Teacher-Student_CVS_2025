import os
import json
import torch
from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader

class SwinDataset():
    def __init__(self, fold, split):
        self.json_file = json.load(open(f'data/Fold{fold}/{split}.json'))
        self.images_info = self.json_file['annotations']
        self.transforms_student = get_transform_sequence(256)
        self.transforms_teacher = get_transform_sequence(384)

        self.images_path = 'Dataset/frames'
    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):

        img_info = self.images_info[idx]
        annot = torch.tensor(img_info['cvs'], dtype=torch.float32)
        image = Image.open(os.path.join(self.images_path, img_info['image_name']))
        image_student = self.transforms_student(image)
        image_student = (image_student-torch.min(image_student)) / (-torch.min(image_student)+torch.max(image_student)) #Normalize the image in the interval (0,1)

        image_teacher = self.transforms_teacher(image)
        image_teacher = (image_teacher-torch.min(image_teacher)) / (-torch.min(image_teacher)+torch.max(image_teacher)) #Normalize the image in the interval (0,1)

        return image_student, annot, image_teacher, img_info['image_name'], 


def get_dataloader(train_dataset, val_dataset, batch):
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader

def get_transform_sequence(size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_sequence = transforms.Compose([   transforms.CenterCrop(480),
                                                transforms.Resize((size, size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=torch.tensor(mean),
                                                    std=torch.tensor(std))])
    return transform_sequence