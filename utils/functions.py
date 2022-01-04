import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
# from skimage import morphology
# from skimage import filters
from utils.imresize import imresize
import os
import random
from sklearn.cluster import KMeans


# custom weights initialization called on netG and netD

def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img, opt.ref_image))
    return np2torch(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


# def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

# def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp, 0, 1)
    return inp


def save_image(real_cpu, receptive_feild, ncs, epoch_num, file_name):
    fig, ax = plt.subplots(1)
    if ncs == 1:
        ax.imshow(real_cpu.view(real_cpu.size(2), real_cpu.size(3)), cmap='gray')
    else:
        # ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0, 0), receptive_feild, receptive_feild, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)


def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(size, device=device)
        # noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def plot_learning_curves(G_loss, D_loss, epochs, label1, label2, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, G_loss, n, D_loss)
    # plt.title('loss')
    # plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1, label2], loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)


def plot_learning_curve(loss, epochs, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # 计算模型梯度
    # 其中的grad_outputs表示后向传播中上层传入的梯度，默认为None，这里设为1其实也是一样的（所以该参数多余）
    # retain_graph并不推荐设为True，大部分情况下有更有效率的做法
    # only_inputs参数则已经过时，会被忽略掉。如果想积累来自其他部分的梯度，可以用backward替代
    # 返回值会是tuple，其中有多个tensor（如果inputs和outputs是很多个tensor的话）
    # 由于这里的inputs和outputs均只有1个，对输出取[0]即可得到梯度tensor
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    # 将权重沿着通道方向求2范数，减一平方后整个输出图像的均值作为对梯度权重的惩罚项
    # LAMBDA则是控制该惩罚项的超参数
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))  # 将图像读为numpy矩阵
    # 但实际上读取到的是4个通道，毕竟是png
    x = np2torch(x, opt)  # 转为torch格式
    x = x[:, 0:3, :, :]  # 仅取前三个通道，忽略Alpha通道
    return x


def read_image_dir(dir, opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def np2torch(x, opt):
    if opt.nc_im == 3:  # 模型输入通道数若为3，则
        x = x[:, :, :, None]  # 创建出空的第4维
        x = x.transpose((3, 2, 0, 1)) / 255  # 转换维tensor格式并归一化
    else:
        x = color.rgb2gray(x)  # 否则视为模型只接受灰度图
        x = x[:, :, None, None]  # 创建两维信息
        x = x.transpose(3, 2, 0, 1)  # 同样转置成tensor格式
    x = torch.from_numpy(x)  # numpy转torch
    if not (opt.not_cuda):
        x = move_to_gpu(x)  # 若有GPU则放到GPU中
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)  # 根据是否有GPU判断应该转换为什么类型
    # x = x.type(torch.FloatTensor)
    x = norm(x)  # 再归一化一下
    return x


def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))


def adjust_scales2image(real_, opt):
    """
    该函数可以设置好金字塔层数opt.stop_scale和原图到第一层的缩小倍数opt.scale1
    :param real_: 输入图像
    :param opt: 参数
    :return: 缩小至最精细尺度的图像
    """
    original_size = np.array(real_.shape[2:4])
    max_edge_len, min_edge_len = [original_size[0], original_size[1]] if original_size[0] > original_size[1] else [
        original_size[1], original_size[0]]
    finest_size = np.ceil(original_size * opt.max_size / max_edge_len)
    coarsest_size = np.ceil(original_size * opt.min_size / min_edge_len)

    opt.pyramid_layer_num = math.ceil(
        math.log((coarsest_size[0] / finest_size[0]).astype(float), opt.scale_factor_init)) + 1
    opt.scale1 = opt.max_size / max_edge_len
    real = imresize(real_, opt.scale1, opt)
    return real


def creat_reals_pyramid(real, opt):
    reals = []
    for i in range(0, opt.scale_num + 1, 1):
        scale = math.pow(opt.scale_factor, opt.scale_num - i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals  # 总之是根据缩放次数生成了尺寸不同的真实图像


def load_trained_pyramid(opt, mode_='train'):
    # dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if (os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs, Zs, reals, NoiseAmp


def generate_in2coarsest(reals, scale_v, scale_h, opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else:  # if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s


def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = './TrainedModels/%s/scale_factor=%f,alpha=%d' % (
            opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    elif (opt.mode == 'animation_train'):
        dir2save = './TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (
            opt.input_name[:-4], opt.scale_factor_init, opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (
            opt.out, opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def calc_init_scale(opt):
    in_scale = math.pow(1 / 2, 1 / 3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))  # 循环次数，这里取了3 * log2(s)。取任意整数其实都可以
    in_scale = pow(opt.sr_factor, 1 / iter_num)  # 单次缩放倍数，论文里有提及
    return in_scale, iter_num


def quant(prev, device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    # x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x, centers


def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    # x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask, opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0, vmax=1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask