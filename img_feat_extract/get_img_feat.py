# Based on following code bases
# https://github.com/libeineu/fairseq_mmt
# --------------------------------------------------------'

import timm
import os
import torch
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from PIL import Image

#image dir name
dic = {
    'test': 'ocrmt30k',
    'train': 'ocrmt30k',
    'valid': 'ocrmt30k',
    }
#Image index
dic1 = {
    'test': 'image_test.txt',
    'train': 'image_train.txt',
    'valid': 'image_valid.txt',
    }

dic2 = {
    'test': 'test',
    'train': 'train',
    'valid': 'valid',
    }

dic_model = [
    'vit_base_patch16_224',
    'vit_tiny_patch16_384',
    'vit_small_patch16_384',
    'vit_base_patch16_384',
    'vit_large_patch16_384',
]

def get_filenames(path):
    l = []
    with open(path, 'r') as f:
        for line in f:
            l.append(line.strip().split('#')[0])
    return l

if __name__ == "__main__":
    # please see scripts/README.md firstly. 
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('--dataset', type=str, choices=['train', 'valid', 'test'], help='which dataset')
    parser.add_argument('--path', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    # /path/ocrmt30k
    img_path = args.path
    dataset = args.dataset
    model_name = args.model
    save_dir = os.path.join('ocrmt30k_img_feature', model_name)
    save_dir = os.path.join(save_dir,dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('extracting ' + dataset + '\'s image feature from '+model_name) 
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to('cuda:0') # if use cpu, uncomment '.to('cuda:0')'
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    tmp = []
    count = 0

    filenames = get_filenames(os.path.join(img_path, dic1[dataset]))
    
    with torch.no_grad():
        for i in tqdm(filenames):
            i = os.path.join(img_path, dic[dataset]+'-images-part', i)
            img = Image.open(i).convert("RGB")
            input = transform(img).unsqueeze(0).to('cuda:0') # transform and add batch dimension
            

            out = model.forward_features(input)
            # out = model.forward(input)
            torch.save(out.squeeze(0).cpu(), os.path.join(save_dir, dic2[dataset] + str(count) + '.pth'))
            count+=1
            