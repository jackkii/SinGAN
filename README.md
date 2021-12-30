# SinGAN
原仓库：https://github.com/tamarott/SinGAN

复现SinGAN论文代码

```shell
SinGAN项目文件结构
├── checkpoints # 存放模型的地方 
├── data        # 训练测试的图片
├── scripts     # 各种训练，测试脚本
├── utils       # 各种工具代码
├── options.py  # 定义各种配置实验参数，以命令行形式传入 
├── eval.py     # 测试代码 
├── loss.py     # loss函数 
├── metrics.py  # 评估指标文件 
├── model.py    # 生成器G模型和判别器D模型 
├── train.py    # 训练代码 
└── README.md   # 项目文档
```

## options.py
定义命令行传入的配置参数，通过argparse库实现
 - 是否使用GPU
    - --not_cuda
 - 网络参数G/D
    - --netG
    - --netD
 - 输入输出图片存储路径
    - --input_dir
    - --input_name
    - --out
 - 代码运行模式
    - --mode
 - 随机种子数
    - --manualSeed
 - 图片/噪声参数
    - --nc_z
    - --nc_im
 - 网络超参数
    - --nfc
    - --min_nfc
    - --num_layer
    - --ker_size
    - --stride
    - --padd_size
 - ground truth 金字塔参数
    - --scale_factor
    - --noise_amp
    - --min_size
    - --max_size
 - 训练超参数
    - --niter
    - --lr_g
    - --lr_d
    - --gamma
    - --beta1
    - --Gsteps
    - --Dsteps
    - --lambda_grad
    - --alpha
