import os.path as osp
import math
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchmetrics import Accuracy
import os
from einops import rearrange

from utils.utils import get_pretrained_models
from models.model_component import ClassificationHead
from models.TimeSformer import TimeSformer
from models.ViViT import ViViT
from models.SwinTransformer import SwinTransformer3D
from models.MViTv2 import MViT

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr,  min_lr=5e-5, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases following the
	values of the cosine function between 0 and `pi * cycles` after a warmup
	period during which it increases linearly between 0 and base_lr.
	"""
	# step means epochs here
	def lr_lambda(current_step):
		current_step += 1
		if current_step <= num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps)) # * base_lr 
		progress = min(float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)), 1)
		factor = 0.5 * (1. + math.cos(math.pi * progress))
		return factor*(1 - min_lr/base_lr) + min_lr/base_lr

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class VideoTransformer(pl.LightningModule):
	def __init__(self, 
				 configs,
				 trainer,
				 ckpt_dir,
				 do_eval,
				 do_test,
				 pretrain=False,
				 n_crops=3):
		super().__init__()
		self.configs = configs
		self.trainer = trainer
		self.pretrain=pretrain

		if pretrain: self.n_class=math.factorial(int(configs.pretrain.cubic_puzzle.n_grid[2]))
		
		get_pretrained_models()
		if self.configs.model.pretrain_type.upper()=='IMAGENET-21K': pretrain_pth=os.path.dirname(os.path.abspath(__file__))+'/models/pretrained/vit_base_patch16_224-21k.pth'
		elif self.configs.model.pretrain_type.upper()=='IMAGENET-1K': pretrain_pth=os.path.dirname(os.path.abspath(__file__))+'/models/pretrained/vit_base_patch16_224-1k.pth'
		else: pretrain_pth=None

		# build models
		if self.configs.model.type.upper() == 'VIVIT':
			self.model = ViViT(
					pretrain_pth=pretrain_pth,
					tube_size=self.configs.model.tubelet_size,
					img_size=self.configs.data.img_size,
					num_frames=self.configs.data.n_frames,
					attention_type='fact_encoder')
			if pretrain:
				self.spatial_cls_head = ClassificationHead(
					self.n_class,self.model.embed_dims*int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),pretrain=True
				)
				self.temporal_cls_head = ClassificationHead(
					self.n_class,self.model.embed_dims*int(configs.pretrain.cubic_puzzle.n_grid[2]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[2]),pretrain=True
				)
			else:
				self.cls_head = ClassificationHead(
					self.configs.data.num_classes, self.model.embed_dims)
			
		elif self.configs.model.type.upper() == 'TIMESFORMER':
				self.model = TimeSformer(
					pretrain_pth=pretrain_pth,
					img_size=self.configs.data.img_size,
					num_frames=self.configs.data.n_frames,
					patch_size=self.configs.model.patch_size,
					attention_type='divided_space_time')
				if pretrain:
					self.spatial_cls_head = ClassificationHead(
						self.n_class,self.model.embed_dims*int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),pretrain=True
					)
					self.temporal_cls_head = ClassificationHead(
						self.n_class,self.model.embed_dims*int(configs.pretrain.cubic_puzzle.n_grid[2]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[2]),pretrain=True
					)
				else:
					self.cls_head = ClassificationHead(
						self.configs.data.num_classes, self.model.embed_dims)
				
		elif self.configs.model.type.upper() == 'SWINTRANSFORMER':
				
				if self.configs.model.pretrain_type.upper()=='IMAGENET-21K': pretrain_pth=os.path.dirname(os.path.abspath(__file__))+'/models/pretrained/swin_base_patch4_window7_224-22k.pth'
				elif self.configs.model.pretrain_type.upper()=='IMAGENET-1K': pretrain_pth=os.path.dirname(os.path.abspath(__file__))+'/models/pretrained/swin_base_patch4_window7_224-1k.pth'
				else: pretrain_pth=None
				
				self.model = SwinTransformer3D(
					pretrained=pretrain_pth,
					pretrained2d=True)
				self.model.init_weights()

				num_features = self.model.embed_dim * 2 ** (self.model.num_layers - 1)
				if pretrain:
					self.spatial_cls_head = ClassificationHead(
						self.n_class,num_features*int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),pretrain=True,mode='swin'
					)
					self.temporal_cls_head = ClassificationHead(
						self.n_class,num_features*int(configs.pretrain.cubic_puzzle.n_grid[2]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[2]),pretrain=True,mode='swin'
					)
				else:
					self.cls_head = ClassificationHead(
						self.configs.data.num_classes, num_features,mode='swin')
					
		elif self.configs.model.type.upper() == 'MVIT':
				
				if self.configs.model.pretrain_type.upper()=='IMAGENET-1K': pretrain_pth=os.path.dirname(os.path.abspath(__file__))+'/models/pretrained/MViTv2_B_in1k.pyth'
				else: pretrain_pth=None
				
				self.model = MViT(
					pretrained=pretrain_pth,
					train_crop_size=self.configs.data.img_size,
					test_crop_size=self.configs.data.img_size,
					n_frames=self.configs.data.n_frames
				)
				if pretrain:
					self.spatial_cls_head = ClassificationHead(
						self.n_class,self.model.embed_dim*int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[0]*configs.pretrain.cubic_puzzle.n_grid[1]),pretrain=True
					)
					self.temporal_cls_head = ClassificationHead(
						self.n_class,self.model.embed_dim*int(configs.pretrain.cubic_puzzle.n_grid[2]),n_cells=int(configs.pretrain.cubic_puzzle.n_grid[2]),pretrain=True
					)
				else:
					self.cls_head = ClassificationHead(
						self.configs.data.num_classes, self.model.embed_dim*8)
				
		else:
			raise Exception("model.type must be chosen from [vivit,timesformer,swintransformer,mvit]")
		
		self.max_top1_acc = 0
		if pretrain:self.train_top1_acc = Accuracy(task="multiclass", num_classes=self.n_class)
		else:self.train_top1_acc = Accuracy(task="multiclass", num_classes=self.configs.data.num_classes)

		if pretrain:self.train_top5_acc = Accuracy(task="multiclass",num_classes=self.n_class,top_k=5)
		else: self.train_top5_acc = Accuracy(task="multiclass",num_classes=self.configs.data.num_classes,top_k=5)

		self.loss_fn = nn.CrossEntropyLoss()
		
		self.train_loss_epoch=[]
		self.valid_loss_epoch=[]

		# common
		self.iteration = 0
		self.data_start = 0
		self.ckpt_dir = ckpt_dir
		self.do_eval = do_eval
		self.do_test = do_test
		if self.do_eval:
			self.val_top1_acc = Accuracy(task="multiclass",num_classes=self.configs.data.num_classes)
			self.val_top5_acc = Accuracy(task="multiclass",num_classes=self.configs.data.num_classes,top_k=5)
		if self.do_test:
			self.n_crops = n_crops
			self.test_top1_acc = Accuracy(task="multiclass",num_classes=self.configs.data.num_classes)
			self.test_top5_acc = Accuracy(task="multiclass",num_classes=self.configs.data.num_classes,top_k=5)
	
	def configure_optimizers(self):
		# build optimzer
		opt=self.configs.train.optimizer.upper()
		lr=self.configs.train.lr
		weight_decay=self.configs.train.weight_decay
		if self.pretrain:
			opt=self.configs.pretrain.optimizer.upper()
			lr=self.configs.pretrain.lr
			weight_decay=self.configs.pretrain.weight_decay

		optimizer = optim.SGD(self.model.parameters(), momentum=0.9, nesterov=True,
							lr=lr, weight_decay=weight_decay)
		if opt == 'ADAMW':
			optimizer = optim.AdamW(self.model.parameters(), betas=(0.9, 0.999),
									lr=lr, weight_decay=weight_decay)
		elif opt == 'ADAM':
			optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.999),
									lr=lr, weight_decay=weight_decay)
			
		if self.configs.train.lr_scheduler == 'multistep':
			lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														  milestones=[5, 11],
														  gamma=0.1)
		elif self.configs.train.lr_scheduler == 'cosine':
			lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
														  num_warmup_steps=self.configs.train.warmup_epoch, 
														  num_training_steps=self.configs.train.epoch,
														  base_lr=self.configs.train.lr,
														  min_lr=1e-5)

		return [optimizer], [lr_scheduler]

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	# epoch schedule
	def _get_momentum(self, base_value, final_value):
		return final_value - (final_value - base_value) * (math.cos(math.pi * self.trainer.current_epoch / self.trainer.max_epochs) + 1) / 2

	def _weight_decay_update(self):
		for i, param_group in enumerate(self.optimizers().optimizer.param_groups):
			if i == 1:  # only the first group is regularized
				param_group["weight_decay"] = self._get_momentum(base_value=self.configs.train.weight_decay, final_value=self.configs.train.weight_decay/10)

	def log_step_state(self, top1_acc=0, top5_acc=0):
		self.log("time",float(f'{time.perf_counter()-self.data_start:.3f}'),prog_bar=True,logger=False)
		
		
	
	# def on_after_backward(self):
	# 	param_norms = self.clip_gradients(self.configs.train.clip_grad)
	# 	self._weight_decay_update()
	# 	# log learning daynamic
	# 	lr = self.optimizers().optimizer.param_groups[0]['lr']
	# 	self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True)

	# Trainer Pipeline
	def training_step(self, batch, batch_idx):
		if self.pretrain:
			spatial_puzzle,temporal_puzzle = batch

			spatial_inputs,spatial_labels=spatial_puzzle
			spatial_inputs=rearrange(spatial_inputs,'B N T C H W -> (B N) T C H W')
			spatial_preds=self.model(spatial_inputs,use_pos_embed=False)
			spatial_preds=self.spatial_cls_head(spatial_preds)
			spatial_loss=self.loss_fn(spatial_preds,spatial_labels.flatten())

			temporal_inputs,temporal_labels=temporal_puzzle
			temporal_inputs=rearrange(temporal_inputs,'B N T C H W -> (B N) T C H W')
			temporal_preds=self.model(temporal_inputs,use_pos_embed=False)
			temporal_preds=self.temporal_cls_head(temporal_preds)
			temporal_loss=self.loss_fn(temporal_preds,temporal_labels.flatten())

			loss=spatial_loss+temporal_loss

			top1_spatial_acc = self.train_top1_acc(spatial_preds.softmax(dim=-1), spatial_labels.flatten())
			top1_temporal_acc = self.train_top1_acc(temporal_preds.softmax(dim=-1), temporal_labels.flatten())

			self.log("spatial_loss",spatial_loss,on_step=True,on_epoch=False,prog_bar=True,logger=True)
			self.log("temporal_loss",temporal_loss,on_step=True,on_epoch=False,prog_bar=True,logger=True)
			self.log("top1_spatial_acc",top1_spatial_acc,on_step=True,on_epoch=False,prog_bar=True,logger=True)
			self.log("top1_temporal_acc",top1_temporal_acc,on_step=True,on_epoch=False,prog_bar=True,logger=True)
		else:
			inputs, labels = batch
			
			preds = self.model(inputs)
			preds = self.cls_head(preds)
			cls_loss = self.loss_fn(preds, labels)
			loss=cls_loss

			top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels)
			top5_acc = self.train_top5_acc(preds.softmax(dim=-1), labels)
			self.log("top1_acc",top1_acc,on_step=True,on_epoch=False,prog_bar=True,logger=True)
			self.log("top5_acc",top5_acc,on_step=True,on_epoch=False,prog_bar=True,logger=True)

		lr = self.optimizers().optimizer.param_groups[0]['lr']
		self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True,logger=False)
		self.log("loss",loss.item(),on_step=True,on_epoch=False,prog_bar=True)
		self.train_loss_epoch.append(loss.item())
		
		return {'loss': loss}
			
	def on_train_epoch_end(self):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
		mean_top1_acc = self.train_top1_acc.compute()
		mean_top5_acc = self.train_top5_acc.compute()
		self.print(f'EPOCH[{self.trainer.current_epoch:02d}/{self.trainer.max_epochs:02d}] - TRAIN ACCURACY -> top1_acc:{mean_top1_acc:.3f},top5_acc:{mean_top5_acc:.3f}')
		
		self.log("TRAIN LOSS (EPOCH)",np.mean(self.train_loss_epoch),on_step=False,on_epoch=True,prog_bar=False)
		self.log("TRAIN ACC TOP1 (EPOCH)",mean_top1_acc,on_step=False,on_epoch=True,prog_bar=False)
		self.log("TRAIN ACC TOP5 (EPOCH)",mean_top5_acc,on_step=False,on_epoch=True,prog_bar=False)
		
		self.train_top1_acc.reset()
		self.train_top5_acc.reset()

		self.train_loss_epoch=[]
		
		# save last checkpoint
		save_path = osp.join(self.ckpt_dir, 'last_checkpoint.pth')
		self.trainer.save_checkpoint(save_path)

	def validation_step(self, batch, batch_indx):
		if self.pretrain: return
		if self.do_eval:
			inputs, labels = batch
			with torch.no_grad():
				preds=self.model(inputs)
				preds = self.cls_head(preds)
				loss = self.loss_fn(preds, labels)

			self.valid_loss_epoch.append(loss.item())

			self.val_top1_acc(preds.softmax(dim=-1), labels)
			self.val_top5_acc(preds.softmax(dim=-1), labels)
			self.data_start = time.perf_counter()
	
	def on_validation_epoch_end(self):
		if self.pretrain: return
		if self.do_eval:
			mean_top1_acc = self.val_top1_acc.compute()
			mean_top5_acc = self.val_top5_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'EPOCH[{self.trainer.current_epoch:02d}/{self.trainer.max_epochs:02d}] - VALID ACCURACY -> top1_acc:{mean_top1_acc:.3f},top5_acc:{mean_top5_acc:.3f}')
			
			self.log("VALIDATION LOSS (EPOCH)",np.mean(self.valid_loss_epoch),on_step=False,on_epoch=True,prog_bar=False)
			self.log("VALIDATION ACC TOP1 (EPOCH)",mean_top1_acc,on_step=False,on_epoch=True,prog_bar=False)
			self.log("VALIDATION ACC TOP5 (EPOCH)",mean_top5_acc,on_step=False,on_epoch=True,prog_bar=False)
			
			self.val_top1_acc.reset()
			self.val_top5_acc.reset()
			self.valid_loss_epoch=[]

			# save best checkpoint
			if mean_top1_acc > self.max_top1_acc:
				save_path = osp.join(self.ckpt_dir,
									 f'{timestamp}_'+
									 f'ep_{self.trainer.current_epoch}_'+
									 f'top1_acc_{mean_top1_acc:.3f}.pth')
				self.trainer.save_checkpoint(save_path)
				self.max_top1_acc = mean_top1_acc
			
	def test_step(self, batch, batch_idx):
		if self.do_test:
			inputs, labels, jigsaw_inputs,temporal_indices,spatial_indices = batch
			preds = self.cls_head(self.model(inputs))
			preds = preds.view(-1, self.n_crops, self.configs.data.num_classes).mean(1)

			self.test_top1_acc(preds.softmax(dim=-1), labels)
			self.test_top5_acc(preds.softmax(dim=-1), labels)
			self.data_start = time.perf_counter()
	
	def on_test_epoch_end(self):
		if self.do_test:
			mean_top1_acc = self.test_top1_acc.compute()
			mean_top5_acc = self.test_top5_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'{timestamp} - Evaluating mean ',
					   f'top1_acc:{mean_top1_acc:.3f}, ',
					   f'top5_acc:{mean_top5_acc:.3f} of current test epoch')
			self.test_top1_acc.reset()
			self.test_top5_acc.reset()
