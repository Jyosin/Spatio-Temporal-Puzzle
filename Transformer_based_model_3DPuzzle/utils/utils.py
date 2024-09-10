import time
import os
import os.path as osp
import yaml
import numpy as np 
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
import urllib.request as req
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytubefix import YouTube
from pytubefix.exceptions import BotDetection
import ffmpeg,json,tqdm,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed

class KineticsDownloader():
	base_url="http://youtube.com/watch?v="
	url_dict={
		'train':"https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
		'valid':"https://s3.amazonaws.com/kinetics/400/annotations/val.csv"
	}

	def __init__(self,mode="train",num_workers=3):
		if not os.path.exists(f"./kinetics-400_{mode}.csv"): req.urlretrieve(self.url_dict[mode],f"./kinetics-400_{mode}.csv")
		
		self.csv_path=f"./kinetics-400_{mode}.csv"
		self.mode=mode
		self.threadpool=ThreadPoolExecutor(max_workers=num_workers)

	def download(self,out_dir):
		with open(self.csv_path,mode="r") as f: lines=f.readlines()
		
		classmap={}
		futures=[]
		for line in lines[1:]:
			line=line.replace("\"","")
			id=line.split(",")[1]
			label=line.split(",")[0]
			start_time=int(line.split(",")[2])
			end_time=int(line.split(",")[3])
			if label not in classmap.keys(): classmap[label]=len(classmap)
			futures.append(self.threadpool.submit(self._download,id,label,start_time,end_time,out_dir))

		success=""
		failed=""
		cnt_failed=0
		for future in tqdm.tqdm(as_completed(futures), total=len(lines[1:]),colour='green'):
			try:
				result = future.result()
				success+=result
			except Exception as e:
				msg=str(e)+"\n"
				msg+=traceback.format_exc()+"\n"
				failed+=msg
				cnt_failed+=1

		with open(f"./download-{self.mode}.log",mode="w") as f:
			f.write(f"{cnt_failed}/{len(lines[1:])} videos failed to download\n")
			f.write("[DOWNLOAD FAILED]\n")
			f.write(failed)
			f.write("\n")
			f.write("[DOWNLOAD SUCCESSFULLY]\n")
			f.write(success)
			f.close()

		with open(f'{out_dir}/{os.path.basename(out_dir)}_classmap.json', 'w') as f: json.dump(classmap, f, indent=2)

		self.threadpool.shutdown(wait=True)
        
	def _download(self,id,label,start_time,end_time,out_dir):
		out_path=os.path.join(out_dir,label)
		os.makedirs(out_path,exist_ok=True)
		try:
			if os.path.exists(out_path+f"/{id}.mp4"): return f"youtube_id ({id}) has already downloaded -> {out_path}/{id}.mp4\n"
			YouTube(self.base_url+id).streams.filter(progressive=True,file_extension='mp4').get_highest_resolution().download(out_path,f"_{id}.mp4")
			inp=ffmpeg.input(out_path+f"/_{id}.mp4")
			video = (
				inp
				.trim(start=start_time, end=end_time)
				.setpts('PTS-STARTPTS')
			)
			audio = (
				inp
                .filter('atrim', start=start_time, end=end_time)
                .filter('asetpts', 'PTS-STARTPTS')
            )
			ffmpeg.output(video,audio,out_path+f"/{id}.mp4").overwrite_output().run(quiet=True)
			os.remove(out_path+f"/_{id}.mp4")
			return f"youtube_id ({id}) has successfully downloaded -> {out_path}/{id}.mp4\n"
		except BotDetection as e:
			return self._download(id,label,start_time,end_time,out_dir)
		except:
			raise Exception(f"Error occured when download from {self.base_url+id}")


def get_pretrained_models(model_dest=os.path.join(os.path.dirname(os.path.dirname(__file__)),'models/pretrained')):
    model_urls={
        'vit_base_patch16_224-1k.pth':'https://drive.google.com/file/d/1QjGpbR8K4Cf4TJaDc60liVhBvPtrc2v4/view?usp=sharing',
        'vit_base_patch16_224-21k.pth':'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth',
		'swin_base_patch4_window7_224-1k.pth':'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
        'swin_base_patch4_window7_224-22k.pth':'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
		'MViTv2_B_in1k.pyth':'https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth'
    }
    
    def progress_print(block_count, block_size, total_size):
        percentage = 100.0 * block_count * block_size / total_size
        # 100より大きいと見た目が悪いので……
        if percentage > 100: percentage = 100
        # バーはmax_bar個で100％とする
        max_bar = 100
        bar_num = int(percentage / (100 / max_bar))
        progress_element = '=' * bar_num
        if bar_num != max_bar:
            progress_element += '>'
        bar_fill = ' ' # これで空のとこを埋める
        bar = progress_element.ljust(max_bar, bar_fill)
        total_size_mb = total_size / (2**20)
        print(
            f'[{bar}] {percentage:.2f}% ( {total_size_mb:.2f}MB )\r',
            end=''
        )

    if not os.path.exists(model_dest): os.makedirs(model_dest)
    try:
        for filename,url in model_urls.items():
            if not os.path.exists(os.path.join(model_dest,filename)):
                print(f"Download pretrained model from {url}")
                req.urlretrieve(url,os.path.join(model_dest,filename),progress_print)
        print("Completed")
    except:
        print("Fail to download pretrained models...")
        exit(0)

class AttrDict():
	def __init__(self, dictionary: dict):
		for key,value in dictionary.items():
			if type(value)==dict: setattr(self,key,AttrDict(value))
			elif type(value)==str and (value[0]=='(' and value[-1]==')'):
				value=tuple([float(e) for e in value[1:-1].split(',')])
				setattr(self,key,value)
			else: setattr(self,key,value)

def load_config(config_path='./config.yml')->AttrDict:
	with open(config_path, 'r') as yml: config = yaml.safe_load(yml)
	return AttrDict(config)

def imsave(img_array,save_path):
    if type(img_array)==torch.Tensor: img_array=img_array.cpu().detach().numpy().transpose(1,2,0)
    if np.max(img_array)==1: img_array=np.clip(img_array * 255, a_min = 0, a_max = 255).astype(np.uint8)
    plt.imshow(img_array)
    plt.savefig(save_path)
    plt.close()

@rank_zero_only
def print_on_rank_zero(content):
	if is_main_process():
		print(content)
	
def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True

def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()

def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()

def is_main_process():
	return get_rank() == 0

def timeit_wrapper(func, *args, **kwargs):
	start = time.perf_counter()
	func_return_val = func(*args, **kwargs)
	end = time.perf_counter()
	return func_return_val, float(f'{end - start:.4f}')

def show_trainable_params(named_parameters):
	for name, param in named_parameters:
		print(name, param.size())

def build_param_groups(model):
	params_no_decay = []
	params_has_decay = []
	params_no_decay_name = []
	params_decay_name = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param) == 1 or name.endswith('.bias'): 
			params_no_decay.append(param)
			params_no_decay_name.append(name)
		else:
			params_has_decay.append(param)
			params_decay_name.append(name)

	param_groups = [
					{'params': params_no_decay, 'weight_decay': 0},
					{'params': params_has_decay},
					]
	print_on_rank_zero(f'params_no_decay_name: {params_no_decay_name} \n params_decay_name: {params_decay_name}')
	return param_groups


def denormalize(data, mean, std):
	"""Denormalize an image/video tensor with mean and standard deviation.

	Args:
		input: Image tensor of size : (H W C).
		mean: Mean for each channel.
		std: Standard deviations for each channel.

	Return:
		Denormalised tensor with same size as input : (H W C).
	"""
	shape = data.shape

	if isinstance(mean, tuple):
		mean = np.array(mean, dtype=float)
		mean = torch.tensor(mean, device=data.device, dtype=data.dtype)

	if isinstance(std, tuple):
		std = np.array(std, dtype=float)
		std = torch.tensor(std, device=data.device, dtype=data.dtype)

	if mean.shape:
		mean = mean[None, :]
	if std.shape:
		std = std[None, :]

	out = (data.contiguous().view(-1, shape[-1]) * std) + mean

	return out.view(shape)


def show_processed_image(imgs, save_dir, mean, std, index=0):
	"""Plot the transformed images into figure and save to disk.
	
	Args:
		imgs: Image tensor of size : (T H W C).
		save_dir: The path to save the images.
		index: The index of current clips.
	"""
	os.makedirs(save_dir, exist_ok=True)
	if not isinstance(imgs[0], list):
		imgs = [imgs]
		
	num_show_clips = 5
	num_rows = len(imgs)
	num_cols = num_show_clips
	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
	for row_idx, row in enumerate(imgs):
		row = row[:num_show_clips]
		for col_idx, img in enumerate(row):
			ax = axs[row_idx, col_idx]
			img = denormalize(img, mean, std).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			#img = img.cpu().numpy().astype(np.uint8)
			ax.imshow(np.asarray(img))
			ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.tight_layout()
	filename = osp.join(save_dir, f'clip_transformed_b{index}.png')
	plt.savefig(filename)