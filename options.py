# options.py
# ==============================
# 获取配置信息（命令行输入的参数）
# ===============================
# Jackkii 2021/12/27
#

# argparse 是python用于解析命令行参数和选项的标准模块
import argparse

# 步骤：1.创建对象 ArgumentParser()  ArgumentParser对象包含将命令行解析成 Python 数据类型所需的全部信息
#       2.添加参数 add_argument()
#       3.解析参数  parse_args() 命令行有多余参数会报错   /
#                  parser.parse_known_args () 返回一个元组，一为参数对象，二为多余参数列表

def get_arguments():
    # =================================== 1.创建对象 ====================================
    parser = argparse.ArgumentParser()

    # =================================== 2.添加参数 ====================================
    # 是否使用GPU
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # 网络G\D参数
    parser.add_argument('--netG', help="path to netG (to continue training)", default='')
    parser.add_argument('--netD', help="path to netD (to continue training)", default='')

    # 输入输出图片存储路径
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--out', help='output folder', default='Output')

    # 代码运行模式
    parser.add_argument('--mode', help='task to be done', default='train')

    # 随机种子数
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # 图片/噪声参数
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)

    # 网络超参数
    parser.add_argument('--nfc', type=int, default=32)                                  # 图像通道数
    parser.add_argument('--min_nfc', type=int, default=32)                              # 最小图像通道数（可能不需要？）
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)    # 网络层数
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)          # 卷积核大小
    parser.add_argument('--stride', help='stride', default=1)                           # 卷积步长
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # ground truth 金字塔参数
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)             # 缩放因子
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)           # 初始加性噪声系数
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)  # 图片最小尺寸
    parser.add_argument('--max_size', type=int, help='image maximal size at the final scale', default=250)   # 图片最大尺寸

    # 训练超参数
    parser.add_argument('--niter', type=int, help='number of epochs to train per scale', default=2000)       # 训练回合
    parser.add_argument('--lr_g', type=float, help='learning rate, default=0.0005', default=0.0005)          # G初始学习率
    parser.add_argument('--lr_d', type=float, help='learning rate, default=0.0005', default=0.0005)          # D初始学习率

    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5', default=0.5)
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)           # 梯度惩罚项权重
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)               # 重建损失权重

    return parser