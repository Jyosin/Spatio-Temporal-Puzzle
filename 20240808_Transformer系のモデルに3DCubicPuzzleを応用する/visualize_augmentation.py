from dataset.data_trainer import DataModule,DataModuleK400
from utils.utils import imsave
import argparse,os,tqdm,shutil
from utils.utils import load_config

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("-config_path",type=str,default="./config.yml",help="学習設定ファイルのパス")
    arg.add_argument("-n_samples",type=int,default=5,help="サンプル数")
    args=load_config(arg.parse_args().config_path)

    tgt_dir="./Visualize_Augmentation"
    if os.path.exists(tgt_dir): shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir,exist_ok=True)
    if args.data.datatype=="Kinetics": pretrain_data_module=DataModuleK400(args,pretrain=args.pretrain.cubic_puzzle.is_valid)
    else: pretrain_data_module=DataModule(args,pretrain=args.pretrain.cubic_puzzle.is_valid)
    pretrain_data_module.setup(None)
    train_set=pretrain_data_module.train_dataloader().dataset

    train_set.cubicpuzzle.transforms.transforms=train_set.cubicpuzzle.transforms.transforms[:-1] # Normalizeを除去
    train_set.transform.transforms=train_set.transform.transforms[:-1] # Normalizeを除去
    
    print(f"[Augmentation出力]:{arg.parse_args().config_path}")
    for i in range(arg.parse_args().n_samples):
        print(f"☆ Sample {i+1:02d} ☆")
        os.makedirs(f"{tgt_dir}/{i+1:02d}/original",exist_ok=True)
        os.makedirs(f"{tgt_dir}/{i+1:02d}/cls_input",exist_ok=True)
        os.makedirs(f"{tgt_dir}/{i+1:02d}/pretrain_input_spatial",exist_ok=True)
        os.makedirs(f"{tgt_dir}/{i+1:02d}/pretrain_input_temporal",exist_ok=True)
        
        orig_video=train_set.get_original_video(i)
        for j,v in tqdm.tqdm(enumerate(orig_video[:64]),leave=False): imsave(v,f"{tgt_dir}/{i+1:02d}/original/{j:03d}.png")
        print(f"original video frames was output           -> {tgt_dir}/{i+1:02d}/original")

        train_set.pretrain=False
        train_set.target_video_len=args.data.n_frames
        train_set.temporal_sample.size=args.data.n_frames*args.data.frame_interval
        inputs,labels=train_set[i]
        for j,v in tqdm.tqdm(enumerate(inputs),leave=False): imsave(v,f"{tgt_dir}/{i+1:02d}/cls_input/{j:03d}.png")
        print(f"input frames for classification was output -> {tgt_dir}/{i+1:02d}/cls_input")

        train_set.pretrain=True
        train_set.target_video_len=int(args.pretrain.cubic_puzzle.jitter_size[2]*args.pretrain.cubic_puzzle.n_grid[2])
        train_set.temporal_sample.size=int(args.pretrain.cubic_puzzle.jitter_size[2]*args.pretrain.cubic_puzzle.n_grid[2]*args.data.frame_interval)
        spatial_inputs,temporal_inputs=train_set[i]
        tensor,labels=spatial_inputs
        for j,v in tqdm.tqdm(enumerate(tensor),leave=False): 
            for k,_v in enumerate(v):
                imsave(_v,f"{tgt_dir}/{i+1:02d}/pretrain_input_spatial/{j:03d}-{k:03d}.png")
        print(f"input frames for pretraining was output    -> {tgt_dir}/{i+1:02d}/pretrain_input_spatial")

        tensor,labels=temporal_inputs
        for j,v in tqdm.tqdm(enumerate(tensor),leave=False): 
            for k,_v in enumerate(v):
                imsave(_v,f"{tgt_dir}/{i+1:02d}/pretrain_input_temporal/{j:03d}-{k:03d}.png")
        print(f"input frames for pretraining was output    -> {tgt_dir}/{i+1:02d}/pretrain_input_temporal")
        print()
    

if __name__=='__main__':main()