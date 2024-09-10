## Transformer系のモデルでUCF101を学習させる

### 概要
VideoTransformer系のモデルでUCF101を学習させるプロジェクト。（参考にしたgitリポジトリは[こちら](https://github.com/mx-mark/VideoTransformer-pytorch)）<br>
現在対応しているVideoTransformer系のモデルは下記の通り。
- [TimeSformer](https://arxiv.org/abs/2102.05095)
- [VideoVisionTransformer](https://arxiv.org/abs/2103.15691)
- [SwinTransformer](https://arxiv.org/abs/2106.13230)
- [MViTv2](https://arxiv.org/abs/2112.01526)

### セットアップ
実行する前に下記手順を参考にして環境構築してください。
1. Condaで直接環境構築する場合<br>
    下記コマンドを順番に実行して環境構築してください。
    ```
    conda create -n env-ViViT python=3.10 -y
    conda activate env-ViViT
    pip install --upgrade pip
    pip install -r requirements.txt
    conda install tensorboard
    ```
2. Dockerで環境構築する場合<br>
    下記手順で環境構築してください。<br>
    1. Dockerfileからイメージ構築
        ```
        docker build . -t img_env-ViViT
        ```
    2. イメージからコンテナ作成
        ```
        docker run -v ./:/work -n env-ViViT -it img_env-ViViT
        ```

3. UCF-101のデータセットがない場合は、下記コマンドでダウンロードしてください。<br>
    ```
    wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
    unrar x UCF101.rar
    ```

### 使い方
- セットアップで環境構築した後で下記でトレーニングできます。学習のハイパーパラメータはconfig.ymlで指定できます。
    ```
    python main.py -config_path config.yml
    ```
- 学習の経過はTensor Boardで確認できます。下記コマンドでTensorBoardを実行できます。
    ```
    tensorboard --logdir [config.ymlのothers → log_pathの設定値]
    ```
    TensorBoardはデフォルトで[http://localhost:6006/](http://localhost:6006/)にアクセスすると見れます。
- 動画のAugmentationは以下を実行することで確認できます。(Visualize_AugmentationフォルダにAugmentation後の画像が出力されます)
    ```
    python visualize_augmentation.py -config_path config.yml
    ```