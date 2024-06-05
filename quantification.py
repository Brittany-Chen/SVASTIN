import argparse
import torch
from torchvision.utils import save_image
import os
import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='number of data loading workers',
                    default="")
args = parser.parse_args()

# 将npy存成png
clean_npy_path = os.path.join(args.root_path, "clean_video")
clean_image_path = os.path.join(args.root_path, 'clean_images')
adv_npy_path = os.path.join(args.root_path, 'adv_video')
adv_image_path = os.path.join(args.root_path, 'adv_images')

if not os.path.exists(clean_image_path):
    os.mkdir(clean_image_path)
if not os.path.exists(adv_image_path):
    os.mkdir(adv_image_path)
count = 0

for path in [clean_npy_path, adv_npy_path]:
    for i in os.listdir(path):
        if i.split(".")[-1] == "npy":
            count = count + 1
            print("count = ", count)
            print(os.path.join(path, i))
            loadData = np.load(os.path.join(path, i))
            print("----type----")
            print(type(loadData))
            print("----shape----")
            print(loadData.shape)
            data_tensor = torch.from_numpy(loadData)  # 1,3,16,224,224
            data_tensor = data_tensor.permute(0, 2, 1, 3, 4)

            print(data_tensor.shape, type(data_tensor))
            print(path, str(i.split(".")[0]))
            print(data_tensor.nelement(), data_tensor.nelement() - torch.count_nonzero(data_tensor / 255.0).item())
            if path == clean_npy_path:
                save_image((data_tensor.squeeze(0) / 255.0),
                           os.path.join(clean_image_path, str(i.split(".")[0] + ".png")))
            else:
                save_image((data_tensor.squeeze(0) / 255.0),
                           os.path.join(adv_image_path, str(i.split(".")[0] + ".png")))


# 第二部分：量化
def l_cal(img1, img2):
    noise = (img1 - img2).flatten(start_dim=0)
    l2 = torch.sum(torch.pow(torch.norm(noise, p=2, dim=0), 2))
    l_inf = torch.sum(torch.norm(noise, p=float('inf'), dim=0))
    return l2, l_inf


transform = T.Compose([
    T.ToTensor(),
])
mse = 0
psnr = 0
ssim = 0
lpipsnum = 0
count = 0
l_2sum = 0
l_infsum = 0
CS = 0
l21 = 0
loss_on_vgg = lpips.LPIPS(net='vgg')


def process_image(img1path, img2path):
    img1 = Image.open(img1path)
    img1_t = transform(img1)
    lp1 = lpips.im2tensor(lpips.load_image(img1path))
    img1 = np.array(img1)
    img2 = Image.open(img2path)
    img2_t = transform(img2)
    lp2 = lpips.im2tensor(lpips.load_image(img2path))
    img2 = np.array(img2)

    print(img1_t.shape)
    l21 = torch.pow((img2_t - img1_t), 2).sum(dim=2).mean(dim=1).sum(dim=0)
    tmp_cos_loss_0 = torch.nn.functional.cosine_similarity(img1_t[0], img2_t[0])
    tmp_cos_loss_1 = torch.nn.functional.cosine_similarity(img1_t[1], img2_t[1])
    tmp_cos_loss_2 = torch.nn.functional.cosine_similarity(img1_t[2], img2_t[2])
    CS = torch.mean(tmp_cos_loss_0 + tmp_cos_loss_1 + tmp_cos_loss_2).item() / 3.0
    mse = compare_mse(img1, img2)
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, channel_axis=2)
    lpipsnum = loss_on_vgg(lp1, lp2)
    l_2, l_inf = l_cal(img1_t, img2_t)
    l_2sum = l_2
    l_infsum = l_inf
    del img1
    del img2
    return mse, psnr, ssim, lpipsnum, l_2sum, l_infsum, CS, l21


for i in range(len(os.listdir(adv_image_path))):
    print(os.listdir(adv_image_path)[i])
    img1path = os.path.join(adv_image_path, os.listdir(adv_image_path)[i])
    img2path = clean_image_path + '\\{}-ori.png'.format(os.listdir(adv_image_path)[i].split('-')[0])
    print(img1path, img2path)
    MSE, PSNR, SSIM, LPIPSUM, L_2SUM, L_INFSUM, cs, L21 = process_image(img1path, img2path)
    mse += MSE
    psnr += PSNR
    ssim += SSIM
    lpipsnum += LPIPSUM
    l_2sum += L_2SUM
    l_infsum += L_INFSUM
    CS += cs
    l21 += L21
    count += 1
print("mse:" + str(mse / count))
print("psnr:" + str(psnr / count))
print("ssim:" + str(ssim / count))
print("lpips:" + str(lpipsnum / count))
print("l_2:" + str(l_2sum / count))
print("l_inf:" + str(l_infsum / count))
print("CS:" + str(CS / count))
print('l21:' + str(l21 / count))
print(count)
