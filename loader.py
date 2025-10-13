from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
import cv2 # Importe o OpenCV
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       
    
class weak_annotation(torch.nn.Module):
    def __init__(self, patch_size = 16, img_size = 256):
        super().__init__()
        self.arranger = Rearrange('c (ph h) (pw w) -> c (ph pw) h w', c=1, h=patch_size, ph=img_size//patch_size, w=patch_size, pw=img_size//patch_size)
    def forward(self, x):
        x = self.arranger(x)
        x = torch.sum(x, dim = [-2, -1])
        x = x/x.max()
        return x
    
def Bextraction(img):
    img = img[0].numpy()
    img2 = binary_dilation(img, structure=np.ones((7,7))).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())

## Temporary
class isic_loader(Dataset):
    """ dataset class for ISIC datasets - Versão com Augmentation Robusto
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self).__init__()
        self.train = train
        if train:
            self.data  = np.load(path_Data+'data_train.npy')
            self.mask  = np.load(path_Data+'mask_train.npy')
        else:
            if Test:
                self.data  = np.load(path_Data+'data_test.npy')
                self.mask  = np.load(path_Data+'mask_test.npy')
            else:
                self.data  = np.load(path_Data+'data_val.npy')
                self.mask  = np.load(path_Data+'mask_val.npy')
        
        # A normalização agora será feita pelo Albumentations, então esta linha pode ser removida
        # self.data  = dataset_normalized(self.data)
        
        self.mask  = np.expand_dims(self.mask, axis=3)
        self.mask  = self.mask / 255.
        
        # ===== NOVO: Definindo os pipelines de transformação =====
        if self.train:
            # Pipeline robusto para TREINAMENTO
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.7),
                A.RandomBrightnessContrast(p=0.5),
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Médias e desvios padrão da ImageNet, um bom ponto de partida
                ToTensorV2(), # Converte para Tensor e ajusta as dimensões
            ])
        else:
            # Pipeline simples para VALIDAÇÃO/TESTE (apenas normalização e conversão)
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
        self.weak_annotation = weak_annotation(patch_size = 16, img_size = 256)

    # ===== NOVO: O método de aumento de dados agora usa o pipeline do Albumentations =====
    def apply_augmentation(self, img, seg):
        # O seg (máscara) vem com uma dimensão extra, vamos removê-la para o Albumentations
        seg = np.squeeze(seg, axis=-1)
        
        # Aplica a transformação
        augmented = self.transform(image=img, mask=seg)
        
        img_aug = augmented['image']
        seg_aug = augmented['mask']
        
        # Adiciona a dimensão do canal na máscara e garante que seja float
        seg_aug = seg_aug.unsqueeze(0).float()
        
        return img_aug, seg_aug

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        
        # ===== MODIFICADO: A lógica de aumento e conversão para tensor agora está centralizada =====
        if self.train:
            # A imagem vem como uint8, precisamos converter para o formato que a transformação espera
            img_aug, seg_aug = self.apply_augmentation(img.astype(np.uint8), seg)
        else:
            # Para validação, aplicamos apenas a normalização e conversão para tensor
            # (que também está dentro do nosso 'else' do self.transform)
            augmented = self.transform(image=img.astype(np.uint8), mask=np.squeeze(seg, axis=-1))
            img_aug = augmented['image']
            seg_aug = augmented['mask'].unsqueeze(0).float()

        # As variáveis agora já são tensores, com o formato correto (C, H, W)
        img = img_aug
        seg = seg_aug
        
        # A lógica abaixo continua a mesma
        weak_ann = self.weak_annotation(seg)
        boundary = Bextraction(seg)

        return {'image': img,
                'weak_ann': weak_ann,
                'boundary': boundary,
                'mask' : seg}

    def __len__(self):
        return len(self.data)
    