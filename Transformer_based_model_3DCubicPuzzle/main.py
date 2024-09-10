import os,shutil
import time
import random
import warnings
import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from utils.utils import load_config,KineticsDownloader
from dataset.data_trainer import DataModule,DataModuleK400

from trainer import VideoTransformer

torch.set_float32_matmul_precision("medium")

def single_run():
	arg=argparse.ArgumentParser()
	arg.add_argument("-config_path",type=str,default="./config.yml",help="学習設定ファイルのパス")
	args=load_config(arg.parse_args().config_path)
	print(f"Config file was loaded from {arg.parse_args().config_path}")

	warnings.filterwarnings('ignore')

	log_dir=args.others.log_path+f"/modeltype_{args.model.type}-pretrain_{args.model.pretrain_type}-optimizer_{args.train.optimizer}-lr_{args.train.lr}-nframe_{args.data.n_frames}-frameinterval_{args.data.frame_interval}-cubic_{args.pretrain.cubic_puzzle.is_valid}-ngrid_{args.pretrain.cubic_puzzle.n_grid}-jittersize_{args.pretrain.cubic_puzzle.jitter_size}"
	ckpt_dir = args.others.ckpt_path
	
	if os.path.exists(log_dir): shutil.rmtree(log_dir)
	
	os.makedirs(ckpt_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	effective_batch_size = args.train.batch_size
	args.train.lr = args.train.lr * effective_batch_size / 256

	effective_batch_size = args.pretrain.batch_size
	args.pretrain.lr = args.pretrain.lr * effective_batch_size / 256

	if args.data.datatype=="Kinetics":
		if not os.path.exists(args.data.Kinetics_train_path):
			print(f"Kinetics400 train dataset was not found on {args.data.Kinetics_train_path}")
			print(f"Start Downloading dataset to {args.data.Kinetics_train_path}")
			loader=KineticsDownloader(num_workers=10)
			loader.download(args.data.Kinetics_train_path)
		if not os.path.exists(args.data.Kinetics_valid_path):
			print(f"Kinetics400 validation dataset was not found on {args.data.Kinetics_valid_path}")
			print(f"Start Downloading dataset to {args.data.Kinetics_valid_path}")
			loader=KineticsDownloader(num_workers=10,mode='valid')
			loader.download(args.data.Kinetics_valid_path)

	# Data
	do_eval = True if args.train.val_ratio>0 else False

	# To be reproducable
	torch.random.manual_seed(args.others.rnd_seed)
	np.random.seed(args.others.rnd_seed)
	random.seed(args.others.rnd_seed)
	pl.seed_everything(args.others.rnd_seed, workers=True)
	
	logger=pl_loggers.TensorBoardLogger(save_dir=log_dir,name=args.model.type)

	if args.pretrain.cubic_puzzle.is_valid:
		print("======================  BEGIN PRETRAIN PHASE ======================")
		if args.data.datatype=="Kinetics": pretrain_data_module = DataModuleK400(configs=args,pretrain=True)
		else: pretrain_data_module = DataModule(configs=args,pretrain=True)

		pretrainer = pl.Trainer(
			accelerator="cuda",
			precision=16,
			max_epochs=args.pretrain.epoch,
			check_val_every_n_epoch=1,
			log_every_n_steps=1,
			logger=logger
		)
		
		# Model
		model = VideoTransformer(configs=args, 
								trainer=pretrainer,
								ckpt_dir=ckpt_dir,
								do_eval=False,
								do_test=False,pretrain=True)
		
		pretrainer.fit(model, pretrain_data_module)
		print("=======================  END PRETRAIN PHASE =======================")

	if args.data.datatype=='Kinetics': data_module = DataModuleK400(configs=args)
	else:data_module = DataModule(configs=args)

	trainer = pl.Trainer(
		accelerator="cuda",
		precision=16,
		max_epochs=args.train.epoch,
		check_val_every_n_epoch=1,
		log_every_n_steps=1,
		logger=logger
	)
	
	# Model
	model = VideoTransformer(configs=args, 
							 trainer=trainer,
							 ckpt_dir=ckpt_dir,
							 do_eval=do_eval,
							 do_test=False)
	
	trainer.fit(model, data_module)
	
if __name__ == '__main__':
	single_run()