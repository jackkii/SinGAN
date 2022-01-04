
# 定义G和D网络
# 待解决问题：for循环生成卷积层的通道数opt.nfc问题
import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    # ConvBlock = cov(3x3)-BN-LeakyRelu(0.2)
    def __init__(self, in_channel, out_channel, ker_size, padding, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padding)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    # 马尔可夫判别器
    def __init__(self, opt, nfc):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        # nfc_current = opt.nfc
        # self.head = ConvBlock(opt.nc_im, nfc_current, opt.ker_size, opt.padd_size, opt.stride)
        self.head = ConvBlock(opt.nc_im, nfc, opt.ker_size, opt.padd_size, opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            # nfc_current = int(opt.nfc/pow(2, (i+1)))
            # block = ConvBlock(2*nfc_current, nfc_current, opt.ker_size, opt.padd_size, opt.stride)
            block = ConvBlock(nfc, nfc, opt.ker_size, opt.padd_size, opt.stride)
            self.body.add_module('block%d'%(i+1), block)
        # self.tail = nn.Conv2d(nfc_current, 1, kernel_size=opt.ker_size, stride=opt.stride, padding=opt.padd_size)
        self.tail = nn.Conv2d(nfc, 1, kernel_size=opt.ker_size, stride=opt.stride, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Generator(nn.Module):
    # 生成器
    def __init__(self, opt, nfc):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.sigmoid = torch.tensor([1]).to(opt.device)
        self.head = ConvBlock(opt.nc_im, nfc, opt.ker_size, opt.padd_size, opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            block = ConvBlock(nfc, nfc, opt.ker_size, opt.padd_size, opt.stride)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(nfc, opt.nc_im, kernel_size=opt.ker_size, stride=opt.stride, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(self.sigmoid * x + y)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)                     # 求出卷积使长边减少的像素数目（宽边与之相同）
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]  # 让y长宽两侧各删去该数目一半的像素，使得x,y尺寸一致
        return x + y
