import os
from options import get_arguments
from utils import functions
from train import *
from utils.manipulate import *
from model.singan import SinGAN

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)  # 根据不同任务，生成不同的结果存储路径

    if (os.path.exists(dir2save)) == False:  # TODO:此处修改了一下便于debug，正式运行时记得删除==False
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real_unsclaed = functions.read_image(opt)  # 获取原图像
        real = functions.adjust_scales2image(real_unsclaed, opt)  # 保证输入图像尺寸在一定范围内

        singan = SinGAN(opt, real)

        train(opt, singan)  # 模型训练
        # SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)       # 生成不同样本
