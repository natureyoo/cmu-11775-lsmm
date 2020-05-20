import os
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torchvision.models as models
import cv2


def extract_frames(video_dir, video_file, output_dir):
    count = 0
    cap = cv2.VideoCapture('{}/{}'.format(video_dir, video_file))   # capturing the video from the given path
    frameRate = 5
    while(cap.isOpened()):
        print('cap opened!')
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            print('break!!')
            break
        if (frameId % frameRate == 0):
            print(count)
            # storing the frames in a new folder named train_1
            filename ='{}/{}_frame{}.jpg'.format(output_dir, video_file.split('.')[0], count)
            count+=1
            cv2.imwrite(filename, frame)
    cap.release()


def extract_features(frame_dir, video_name, model, data_transform, output_dir):
    img_list = [f for f in os.listdir(frame_dir) if f.startswith('{}_'.format(video_name)) and (f.endswith('0.jpg') or f.endswith('5.jpg'))]
    input_list = []
    for img in img_list:
        input_img = Image.open(os.path.join(frame_dir, img))
        input_list.append(data_transform(input_img))
    feature = model(torch.stack(input_list))
    np.save('{}/{}.npy'.format(output_dir, video_name), feature.reshape(feature.shape[0], -1).cpu().detach().numpy())


class ResNet34Inter(nn.Module):
    def __init__(self, original_model):
        super(ResNet34Inter, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)

        return x


def main():
	video_dir = 'downsampled_videos'
	frame_dir = 'frames'
	feature_dir = 'cnn2'

	######################## extracting frames from all videos #################################
	video_list = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
	for idx, video in enumerate(video_list):
	    if idx % 100 == 0:
	        print(idx)
	    extract_frames(video_dir, video, frame_dir)


	######################## extracting features from image frames using resnet34 ##############
	data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

	resnet_model = models.resnet34(pretrained=True)
	res34_inter = ResNet34Inter(resnet_model)


	for param in res34_except1.parameters():
	    param.requires_grad = False
	    

	for data_type in ['train', 'val', 'test']:
		video_list = [f.strip() for f in open('list/{}.video'.format(data_type)).readlines()]
		for idx, video in enumerate(video_list):
		    try:
		        extract_features(frame_dir, video, res34_inter, data_transform, feature_dir)
		        print('complete feature extraction on {}th video {}'.format(idx, video))
		    except:
		        print('Empy Tensor!')


if __name__ == '__main__':
	main()
