import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from dataset.dataset import DataSet,split_dataset
import dataset.transforms as T

class Collator(object):
	def __init__(self):
		pass
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		jigsaw_inputs_list=[]
		temporal_indices=[]
		spatial_indices=[]

		for record in minibatch:
			image_list.append(record[0])
			label_list.append(record[1])
			jigsaw_inputs_list.append(record[2])
			if record[3] is not None: temporal_indices.append(record[3])
			if record[4] is not None: spatial_indices.append(record[4])
			
		minibatch = []
		minibatch.append(torch.stack(image_list))
		
		label = np.stack(label_list)
		minibatch.append(torch.from_numpy(label))

		minibatch.append(torch.stack(jigsaw_inputs_list))

		if len(temporal_indices)!=0: minibatch.append(torch.stack(temporal_indices))
		else: minibatch.append(torch.empty(0))

		if len(spatial_indices)!=0: minibatch.append(torch.stack(spatial_indices))
		else: minibatch.append(torch.empty(0))
		
		return minibatch

class DataModule(pl.LightningDataModule):
	def __init__(self, configs,pretrain=False):
		super().__init__()
		self.configs = configs
		self.pretrain=pretrain

	def setup(self, stage):
		
		color_jitter = self.configs.train.color_jitter if len(self.configs.train.color_jitter.split(','))>1 else None
		scale = None
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		
		train_transform = T.create_video_transform(
			input_size=self.configs.data.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			interpolation='bicubic',
			mean=mean,
			std=std
		)
		
		valid_transform = T.create_video_transform(
				input_size=self.configs.data.img_size,
				is_training=False,
				interpolation='bicubic',
				mean=mean,
				std=std
		)
		
		temporal_sample = T.TemporalRandomCrop(
			self.configs.data.n_frames * self.configs.data.frame_interval)
		
		self.train_dataset,self.val_dataset=split_dataset(
			config=self.configs,
			transforms=(train_transform,valid_transform),
			temporal_samples=(temporal_sample,temporal_sample),
			pretrain=self.pretrain
		)

	def train_dataloader(self):
		if self.pretrain:
			return DataLoader(
				self.train_dataset,
				batch_size=self.configs.pretrain.batch_size,
				num_workers=self.configs.pretrain.num_workers,
				shuffle=True,
				drop_last=True,
				pin_memory=True
			) 
		else:
			return DataLoader(
				self.train_dataset,
				batch_size=self.configs.train.batch_size,
				num_workers=self.configs.train.num_workers,
				collate_fn=Collator().collate,
				shuffle=False,
				drop_last=True,
				pin_memory=True
			)
	
	def val_dataloader(self):
		if self.pretrain:
			if self.configs.train.val_ratio >0 :
				return DataLoader(
					self.val_dataset,
					batch_size=self.configs.pretrain.batch_size,
					num_workers=self.configs.pretrain.num_workers,
					shuffle=False,
					drop_last=False,
				)
		else:
			if self.configs.train.val_ratio >0 :
				return DataLoader(
					self.val_dataset,
					batch_size=self.configs.train.batch_size,
					num_workers=self.configs.train.num_workers,
					collate_fn=Collator().collate,
					shuffle=False,
					drop_last=False,
				)
			
class DataModuleK400(pl.LightningDataModule):
	def __init__(self, configs,pretrain=False):
		super().__init__()
		self.configs = configs
		self.pretrain=pretrain

	def setup(self, stage):
		color_jitter = self.configs.train.color_jitter if len(self.configs.train.color_jitter.split(','))>1 else None
		scale = None
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		
		train_transform = T.create_video_transform(
			input_size=self.configs.data.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			interpolation='bicubic',
			mean=mean,
			std=std
		)
		
		valid_transform = T.create_video_transform(
				input_size=self.configs.data.img_size,
				is_training=False,
				interpolation='bicubic',
				mean=mean,
				std=std
		)
		
		temporal_sample = T.TemporalRandomCrop(
			self.configs.data.n_frames * self.configs.data.frame_interval)
		
		self.train_dataset=DataSet(configs=self.configs,transform=train_transform,temporal_sample=temporal_sample,pretrain=self.pretrain,train=True)
		self.val_dataset=DataSet(configs=self.configs,transform=valid_transform,temporal_sample=temporal_sample,pretrain=self.pretrain,train=False)

	def train_dataloader(self):
		if self.pretrain:
			return DataLoader(
				self.train_dataset,
				batch_size=self.configs.pretrain.batch_size,
				num_workers=self.configs.pretrain.num_workers,
				shuffle=True,
				drop_last=True,
				pin_memory=True
			) 
		else:
			return DataLoader(
				self.train_dataset,
				batch_size=self.configs.train.batch_size,
				num_workers=self.configs.train.num_workers,
				#collate_fn=Collator().collate,
				shuffle=True,
				drop_last=True,
				pin_memory=True
			)
	
	def val_dataloader(self):
		if self.pretrain:
			if self.configs.train.val_ratio >0 :
				return DataLoader(
					self.val_dataset,
					batch_size=self.configs.pretrain.batch_size,
					num_workers=self.configs.pretrain.num_workers,
					shuffle=False,
					drop_last=False,
				)
		else:
			if self.configs.train.val_ratio >0 :
				return DataLoader(
					self.val_dataset,
					batch_size=self.configs.train.batch_size,
					num_workers=self.configs.train.num_workers,
					#collate_fn=Collator().collate,
					shuffle=False,
					drop_last=False,
				)