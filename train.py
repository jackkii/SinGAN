import utils.functions as functions
import model as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from utils.imresize import imresize


def train(opt, singan):
    input_previous_gen = torch.zeros(singan.size_pyramid[0])
    for pyramid_layer_ind in range(0, opt.pyramid_layer_num):
        for epoch in range(1, opt.niter):
            input_previous_gen, sigmoid = singan.update_single_layer(pyramid_layer_ind, input_previous_gen, opt)
            print('epoch', epoch)
        print('layer', pyramid_layer_ind, ' done')

    # real_ = functions.read_image(opt)  # 读取图像
    # in_s = 0
    # real = imresize(real_, opt.scale1, opt)  # 获取输入图像
    # reals = functions.creat_reals_pyramid(real, reals, opt)  # 生成ground truth金字塔
    # nfc_prev = 0
    #
    # for scale_num in range(0, opt.scale_num):  # 对金字塔的每级进行opt.iter_num次训练，从最粗糙到最精细
    #     opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)  # 原论文：每放大4次，就使通道数提高两倍。这里还额外限制通道数不大于128
    #
    #     opt.out_ = functions.generate_dir2save(opt)  # 生成要保存图像的文件夹
    #     opt.outf = '%s/%d' % (opt.out_, scale_num)  # 生成文件名
    #     try:
    #         os.makedirs(opt.outf)
    #     except OSError:
    #         pass
    #     # 保存真实金字塔图像
    #     plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
    #
    #     # 初始化模型
    #     D_curr, G_curr = init_models(opt)
    #     if (nfc_prev == opt.nfc):  # 如果通道数比起上一次层并没有增加，则使用上一层的训练参数
    #         G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
    #         D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
    #
    #     z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)
    #
    #     # 梯度清零
    #     G_curr = functions.reset_grads(G_curr, False)
    #     G_curr.eval()
    #     D_curr = functions.reset_grads(D_curr, False)
    #     D_curr.eval()
    #
    #     Gs.append(G_curr)  # Gs是生成器金字塔
    #     Zs.append(z_curr)  # Zs则记录各层噪声
    #     NoiseAmp.append(opt.noise_amp)  # 记录当前噪声倍数
    #
    #     torch.save(Zs, '%s/Zs.pth' % (opt.out_))
    #     torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    #     torch.save(reals, '%s/reals.pth' % (opt.out_))
    #     torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
    #
    #     scale_num += 1
    #     nfc_prev = opt.nfc  # 更新上层输入通道数
    #     del D_curr, G_curr
    print('training complete')
    return


# def train_single_scale(netD, netG, reals, Gs, Zs, in_s, opt, centers=None):
#     real = reals[len(Gs)]
#     # 根据图像求噪声宽高
#     opt.nzx = real.shape[2]
#     opt.nzy = real.shape[3]
#     opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride  # 计算感受野
#
#     if opt.mode == 'animation_train':
#         # 该模式下特意设置了噪声图的padding
#         opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
#         opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
#         pad_noise = 0
#
#     alpha = opt.alpha  # 重建损失的系数
#
#     fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy],
#                                            device=opt.device)  # 生成噪声（有多种噪声可选）,参数为size(长度为3的list)和device
#     z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)  # 以0填充为fixed_noise尺寸相同的矩阵
#
#     # setup optimizer
#     optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
#     schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
#     schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)
#
#     errD2plot = []
#     errG2plot = []
#     D_real2plot = []
#     D_fake2plot = []
#     z_opt2plot = []
#
#     for epoch in range(opt.niter):
#         if (Gs == []) & (opt.mode != 'SR_train'):  # 如果当前训练的是G_N层
#             # 先按通道数为1生成噪声矩阵
#             z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
#             # 再利用expand，复制成3个通道数，再padding
#             z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
#             # noise_与z_opt的生成方式相同
#             noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
#             noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
#         else:
#             # 如果当前要训练的层数小于N，则噪声不要求通道复制，可直接生成
#             noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#             noise_ = m_noise(noise_)
#
#         ############################
#         # (1) Update D network: maximize D(x) + D(G(z))
#         ###########################
#         for j in range(opt.Dsteps):  # 为D网络更新3次
#             # train with real
#             netD.zero_grad()
#
#             # 马尔可夫判别器体现在这个netD是卷积网络，输出是幅图像
#             # 真假概率为输出图像的均值
#             output = netD(real).to(opt.device)
#             # D_real_map = output.detach()
#             # 希望real输出越小越好，故加负号
#             errD_real = -output.mean()  # -a
#             errD_real.backward(retain_graph=True)
#             D_x = -errD_real.item()
#
#             # train with fake
#             if (j == 0) & (epoch == 0):  # 如果是第一个epoch的第一次循环
#                 if (Gs == []) & (opt.mode != 'SR_train'):  # 如果层数为N
#                     # 对于第一次训练，上层生成结果prev全取0
#                     # 用0填充，创建矩阵
#                     prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
#                     in_s = prev  # 注意in_s是未经过padding的
#                     # 进行padding
#                     prev = m_image(prev)
#                     z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
#                     z_prev = m_noise(z_prev)
#                     opt.noise_amp = 1
#                 elif opt.mode == 'SR_train':
#                     z_prev = in_s
#                     criterion = nn.MSELoss()
#                     RMSE = torch.sqrt(criterion(real, z_prev))
#                     opt.noise_amp = opt.noise_amp_init * RMSE
#                     z_prev = m_image(z_prev)
#                     prev = z_prev
#                 else:  # 如果当前训练层数小于N
#                     prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
#                     prev = m_image(prev)
#                     z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
#                     criterion = nn.MSELoss()
#                     RMSE = torch.sqrt(criterion(real, z_prev))
#                     opt.noise_amp = opt.noise_amp_init * RMSE
#                     z_prev = m_image(z_prev)
#             else:
#                 # 在训练过程中，draw_concat都只是简单的返回in_s，由于这段循环内训练的是同一层，故不需要改变in_s
#                 # 因此本段代码只是再次给prev做了一次padding
#                 prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
#                 prev = m_image(prev)
#
#             if opt.mode == 'paint_train':
#                 prev = functions.quant2centers(prev, centers)
#                 plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
#
#             if (Gs == []) & (opt.mode != 'SR_train'):
#                 noise = noise_  # 对于第一次训练，使用生成的噪声
#             else:
#                 noise = opt.noise_amp * noise_ + prev
#             # detach函数返回的tensor，已经从梯度图剥离出来了，无法求梯度
#             # 指向的是同一片内存，如果进行in-place操作，会报错
#             fake = netG(noise.detach(), prev)  # 以噪声作为噪声输入，以0作为上层输入
#             output = netD(fake.detach())
#             # 这里希望fake输出越大越好
#             errD_fake = output.mean()
#             errD_fake.backward(retain_graph=True)
#             D_G_z = output.mean().item()
#
#             gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
#             gradient_penalty.backward()
#
#             # 这才是完整的判别器损失，但我不太理解它为什么不一次性backward，非要分开
#             errD = errD_real + errD_fake + gradient_penalty
#             optimizerD.step()
#
#         errD2plot.append(errD.detach())  # 记录判别器损失
#
#         ############################
#         # (2) Update G network: maximize D(G(z))
#         ###########################
#
#         for j in range(opt.Gsteps):  # 再为生成器更新3次
#             netG.zero_grad()
#             output = netD(fake)
#             # D_fake_map = output.detach()
#             errG = -output.mean()
#             errG.backward(retain_graph=True)  # 对应Adversarial loss中生成网络的部分
#             if alpha != 0:
#                 loss = nn.MSELoss()
#                 if opt.mode == 'paint_train':
#                     z_prev = functions.quant2centers(z_prev, centers)
#                     plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
#                 Z_opt = opt.noise_amp * z_opt + z_prev  # 此时的z_prev全0，opt.noise_amp为1
#                 rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)  # 以噪声作为噪声输入，以0作为上层输入。对应reconstruction loss
#                 rec_loss.backward(retain_graph=True)
#                 rec_loss = rec_loss.detach()
#             else:
#                 Z_opt = z_opt
#                 rec_loss = 0
#
#             optimizerG.step()
#
#         errG2plot.append(errG.detach() + rec_loss)  # 保存生成网络误差
#         D_real2plot.append(D_x)  # 保存判别器网络对真实图像的概率判断
#         D_fake2plot.append(D_G_z)  # 保存判别器网络对生成图像的概率判断
#         z_opt2plot.append(rec_loss)  # 保存Reconstruction loss
#
#         if epoch % 25 == 0 or epoch == (opt.niter - 1):
#             print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
#
#         if epoch % 500 == 0 or epoch == (opt.niter - 1):
#             plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
#             plt.imsave('%s/G(z_opt).png' % (opt.outf),
#                        functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
#             # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
#             # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
#             # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
#             # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
#             # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
#             # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
#             torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
#
#         schedulerD.step()
#         schedulerG.step()
#
#     functions.save_networks(netG, netD, z_opt, opt)
#     # 返回值是最后一次的噪声图z_opt（经过padding）,全0矩阵in_s，和训练好的生成器网络netG
#     return z_opt, in_s, netG


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s
    # 如果当前在训练第N层，Gs为空，因此本函数无用
    if len(Gs) > 0:  # 如果当前训练层数小于N
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                # 对当前已经训练好的层进行遍历（zip遍历会以最短的进行截取）
                if count == 0:  # 对于最底层
                    # 按照未padding前的尺寸生成噪声，并要求沿通道复制
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                                 device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:  # 其余层的噪声则没那么多要求
                    z = functions.generate_noise(
                        [opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]  # 按照real_curr的尺寸截取G_z。实际上两者尺寸应当相等。大概只是防患于未然
                G_z = m_image(G_z)
                # 在将噪声输入生成网络之前，按照noise_amp叠加了上层生成图像进去。按照论文来说,noise_amp应正比于上层生成图像升采样后与当前层真实图像的RMSE
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                # if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


# def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
#     in_s = torch.full(reals[0].shape, 0, device=opt.device)
#     scale_num = 0
#     nfc_prev = 0
#
#     while scale_num < opt.scale_num + 1:
#         if scale_num != paint_inject_scale:
#             scale_num += 1
#             nfc_prev = opt.nfc
#             continue
#         else:
#             opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#             opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#
#             opt.out_ = functions.generate_dir2save(opt)
#             opt.outf = '%s/%d' % (opt.out_, scale_num)
#             try:
#                 os.makedirs(opt.outf)
#             except OSError:
#                 pass
#
#             # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
#             # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
#             plt.imsave('%s/in_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
#
#             D_curr, G_curr = init_models(opt)
#
#             z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num + 1], Gs[:scale_num],
#                                                       Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)
#
#             G_curr = functions.reset_grads(G_curr, False)
#             G_curr.eval()
#             D_curr = functions.reset_grads(D_curr, False)
#             D_curr.eval()
#
#             Gs[scale_num] = G_curr
#             Zs[scale_num] = z_curr
#             NoiseAmp[scale_num] = opt.noise_amp
#
#             torch.save(Zs, '%s/Zs.pth' % (opt.out_))
#             torch.save(Gs, '%s/Gs.pth' % (opt.out_))
#             torch.save(reals, '%s/reals.pth' % (opt.out_))
#             torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
#
#             scale_num += 1
#             nfc_prev = opt.nfc
#         del D_curr, G_curr
#     return


def init_models(opt):
    # generator initialization:
    netG = models.Generator(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
