import json
import random

import decord
import numpy as np
import torch
import os,glob
from pathlib import Path

from einops import rearrange
from skimage.feature import hog
from dataset.transforms import CubicPuzzle3D

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
	"""
	Given the start and end frame index, sample num_samples frames between
	the start and end with equal interval.
	Args:
		frames (tensor): a tensor of video frames, dimension is
			`num video frames` x `channel` x `height` x `width`.
		start_idx (int): the index of the start frame.
		end_idx (int): the index of the end frame.
		num_samples (int): number of frames to sample.
	Returns:
		frames (tersor): a tensor of temporal sampled video frames, dimension is
			`num clip frames` x `channel` x `height` x `width`.
	"""
	index = torch.linspace(start_idx, end_idx, num_samples)
	index = torch.clamp(index, 0, frames.shape[0] - 1).long()
	frames = torch.index_select(frames, 0, index)
	return frames


def numpy2tensor(x):
	return torch.from_numpy(x)

def extract_hog_features(image):
	hog_features_r = hog(image[:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_g = hog(image[:,:,1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_b = hog(image[:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False) #visualize=True
	hog_features = np.concatenate([hog_features_r,hog_features_g,hog_features_b], axis=-1)
	hog_features = rearrange(hog_features, '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)', ph=14, pw=14)
	return hog_features

def load_annotation_data(data_file_path):
	with open(data_file_path, 'r') as data_file:
		return json.load(data_file)

def get_class_labels(anno_pth):
	global class_labels_map, cls_sample_cnt
	
	if class_labels_map is not None:
		return class_labels_map, cls_sample_cnt
	else:
		cls_sample_cnt = {}
		class_labels_map = load_annotation_data(anno_pth)
		for cls in class_labels_map:
			cls_sample_cnt[cls] = 0
		return class_labels_map, cls_sample_cnt

def load_annotations(ann_file, num_class, num_samples_per_cls,label_type):
	dataset = []
	class_to_idx, cls_sample_cnt = get_class_labels("./"+label_type+"_classmap.json")
	with open(ann_file, 'r') as fin:
		for line in fin:
			line_split = line.strip().split('\t')
			sample = {}
			idx = 0
			# idx for frame_dir
			frame_dir = line_split[idx]
			sample['video'] = frame_dir
			idx += 1
								
			# idx for label[s]
			label = [x for x in line_split[idx:]]
			assert label, f'missing label in line: {line}'
			assert len(label) == 1
			class_name = label[0]
			class_index = int(class_to_idx[class_name])
			
			# choose a class subset of whole dataset
			if class_index < num_class:
				sample['label'] = class_index
				if cls_sample_cnt[class_name] < num_samples_per_cls:
					dataset.append(sample)
					cls_sample_cnt[class_name]+=1

	return dataset


class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=1, **kwargs):
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		self.kwargs = kwargs
		
	def __call__(self, filename):
		"""Perform the Decord initialization.
		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		reader = decord.VideoReader(filename,
									ctx=self.ctx,
									num_threads=self.num_threads)
		return reader

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'sr={self.sr},'
					f'num_threads={self.num_threads})')
		return repr_str

class DataSet(torch.utils.data.Dataset):
	"""Load the Kinetics video files
	
	Args:
		annotation_path (string): Annotation file path.
		num_class (int): The number of the class.
		num_samples_per_cls (int): the max samples used in each class.
		target_video_len (int): the number of video frames will be load.
		align_transform (callable): Align different videos in a specified size.
		temporal_sample (callable): Sample the target length of a video.
	"""

	def __init__(self,
				 configs,
				 data=None,
				 transform=None,
				 temporal_sample=None,
				 pretrain=False,
				 train=False
				 ):
		
		self.configs = configs

		if configs.data.datatype=="Kinetics":
			if train==True: data_path=self.configs.data.Kinetics_train_path
			else: data_path=self.configs.data.Kinetics_valid_path
			label_type=os.path.basename(data_path)
			data_pathes=sorted(glob.glob(data_path+"/*/*"+configs.data.movie_ext))

			class_to_idx=load_annotation_data(data_path+"/"+label_type+"_classmap.json")
			data_dicts=[]
			for path in data_pathes:
				class_name=Path(path).parent.name
				class_idx=int(class_to_idx[class_name])
				data_dicts.append({'video':path,'label':class_idx})
			self.data=data_dicts
		else:
			if data is None: self.data = load_annotations(self.configs.data.data_path,self.configs.data.movie_ext)
			else: self.data=data

		self.transform = transform
		self.pretrain=pretrain
		
		self.temporal_sample = temporal_sample
		self.target_video_len = self.configs.data.n_frames
		self.v_decoder = DecordInit()

		if pretrain: 
			self.cubicpuzzle=CubicPuzzle3D(
				img_size=configs.data.img_size,
				n_frames=configs.data.n_frames,
				n_grid=configs.pretrain.cubic_puzzle.n_grid,
				grayscale_prob=configs.pretrain.cubic_puzzle.grayscale_prob,
				jitter_size=configs.pretrain.cubic_puzzle.jitter_size,
				crop_mode=configs.pretrain.cubic_puzzle.crop_mode
			)
			self.temporal_sample.size=int(configs.pretrain.cubic_puzzle.jitter_size[2]*configs.pretrain.cubic_puzzle.n_grid[2]*configs.data.frame_interval)
			self.target_video_len=int(configs.pretrain.cubic_puzzle.jitter_size[2]*configs.pretrain.cubic_puzzle.n_grid[2])

	def __getitem__(self, index):
		while True:
			try:
				path = self.data[index]['video']
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				
				start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
				assert end_frame_ind-start_frame_ind >= self.target_video_len
				frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
				video = v_reader.get_batch(frame_indice).asnumpy()
				del v_reader
				break
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)
		
		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0,3,1,2)
			if self.pretrain:
				spatial_puzzle,temporal_puzzle=self.cubicpuzzle(video)
				return spatial_puzzle,temporal_puzzle
			else:
				video = self.transform(video)
				label = self.data[index]['label']
				return video,label

	def __len__(self):
		return len(self.data)
	
	def get_original_video(self,index):
		while True:
			try:
				path = self.data[index]['video']
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				video = v_reader.get_batch(np.arange(total_frames)).asnumpy()
				del v_reader
				break
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)

		return torch.from_numpy(video).permute(0,3,1,2)
	
def split_dataset(config,transforms,temporal_samples,pretrain):
	label_type=os.path.basename(config.data.data_path)
	data_pathes=sorted(glob.glob(config.data.data_path+"/*/*"+config.data.movie_ext))

	class_to_idx,_ = get_class_labels(config.data.data_path+"/"+label_type+"_classmap.json")

	data_dicts=[]
	for path in data_pathes:
		class_name=Path(path).parent.name
		class_idx=int(class_to_idx[class_name])
		data_dicts.append({'video':path,'label':class_idx})

	np.random.shuffle(data_dicts)

	num_trains=int(len(data_dicts)*(1-config.train.val_ratio))
	train_data=data_dicts[:num_trains]
	val_data=data_dicts[num_trains:]

	train_transforms,val_transforms=transforms
	train_temporal_sample,val_temporal_sample=temporal_samples

	return DataSet(config,train_data,train_transforms,train_temporal_sample,pretrain),DataSet(config,val_data,val_transforms,val_temporal_sample,pretrain)