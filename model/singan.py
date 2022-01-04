import math
import torch
import torch.optim as optim
from torch.nn import MSELoss
from utils.imresize import imresize, imresize_to_shape
from utils.functions import generate_noise, calc_gradient_penalty
from model.basic_model import *


class SinGAN():
    def __init__(self, opt, real):
        self.G_pyramid = self._create_G_pyramid(opt)
        self.D_pyramid = self._create_D_pyramid(opt)
        self.real_pyramid, self.size_pyramid = self._create_real_pyramid(opt, real)
        self.init_model(opt)
        self.errG_rec_coarest_input = generate_noise(self.size_pyramid[0])

    def _create_G_pyramid(self, opt):
        G_pyramid = []
        nfc = opt.nfc
        for pyramid_layer_ind in range(0, opt.pyramid_layer_num):
            if pyramid_layer_ind % 4 == 0 and pyramid_layer_ind != 0:
                nfc = min([2 * nfc, 128])
            G = Generator(opt, nfc)
            G_pyramid.append(G)
        return G_pyramid

    def _create_D_pyramid(self, opt):
        D_pyramid = []
        nfc = opt.nfc
        for pyramid_layer_ind in range(0, opt.pyramid_layer_num):
            if pyramid_layer_ind % 4 == 0 and pyramid_layer_ind != 0:
                nfc = min([2 * nfc, 128])
            D = WDiscriminator(opt, nfc)
            D_pyramid.append(D)
        return D_pyramid

    def _create_real_pyramid(self, opt, real):
        reals = []
        sizes = []
        for i in range(0, opt.pyramid_layer_num):
            scale = math.pow(opt.scale_factor, opt.pyramid_layer_num - i)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
            sizes.append(curr_real.shape)
        return reals, sizes

    def init_model(self, opt):
        for i in range(0, opt.pyramid_layer_num):
            self.G_pyramid[i].apply(weights_init)
            self.D_pyramid[i].apply(weights_init)

    def update_single_layer(self, layer_ind, input_previous_gen, opt):
        netG = self.G_pyramid[layer_ind]
        netD = self.D_pyramid[layer_ind]

        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

        input_noise = generate_noise(self.size_pyramid[layer_ind])

        real = self.real_pyramid[layer_ind]
        if torch.cuda.is_available():
            netG = netG.cuda()
            netD = netD.cuda()
            input_noise = input_noise.to(opt.device)
            real = real.to(opt.device)
            input_previous_gen = input_previous_gen.to(opt.device)

        input_previous_gen = imresize_to_shape(input_previous_gen, self.size_pyramid[layer_ind], opt)
        fake = netG(input_noise, input_previous_gen)

        for update_step_ind in range(opt.Dsteps):
            netD.zero_grad()

            real_D_output = netD(real).to(opt.device)
            errD_real = -real_D_output.mean()

            fake_D_output = netD(fake.detach()).to(opt.device)
            errD_fake = fake_D_output.mean()

            gradient_penalty = calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)

            errD = errD_fake + errD_real + gradient_penalty
            errD.backward()
            optimizerD.step()

        for update_step_ind in range(opt.Gsteps):
            netG.zero_grad()
            fake = netG(input_noise, input_previous_gen)
            fake_D_output = netD(fake).to(opt.device)
            errG_adver = -fake_D_output.mean()

            input_rec_noise = torch.zeros(self.size_pyramid[layer_ind]) \
                if layer_ind > 0 else self.errG_rec_coarest_input
            if torch.cuda.is_available():
                input_rec_noise = input_rec_noise.to(opt.device)
            rec_G_output = netG(input_rec_noise, input_previous_gen)
            MSE = MSELoss()
            mse = MSE(rec_G_output, real)
            errG_rec = opt.alpha * mse

            errG = errG_adver + errG_rec
            errG.backward()
            optimizerG.step()
        return fake, mse


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
