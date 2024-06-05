import os
import os.path as osp
import time
import warnings

import numpy as np
from torch.autograd import Variable

from args import get_args_parser
from mmaction.apis import init_recognizer, load_attack_video
from model.model import *
from torch_dwt.torch_dwt.functional import idwt3, dwt3

warnings.filterwarnings("ignore")
args = get_args_parser()
device = torch.device(args.device)

#####################
# Model initialize: #
#####################
model = init_recognizer(args.config_file, args.checkpoint_file, device=device)  # or device='cuda:0'
for p in model.parameters():
    if p.requires_grad:
        p.requires_grad = False
model.eval()
INN_net = Model().to(device)
init_model(INN_net)

total_num = sum(p.numel() for p in INN_net.parameters())
trainable_num = sum(p.numel() for p in INN_net.parameters() if p.requires_grad)
print({'Total': total_num, 'Trainable': trainable_num})

params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
optim_init = optim1.state_dict()
if args.dataset == 'Kinetic400':
    with open("kinetics400_val_list_videos.txt", 'r') as g:
        video_file_val = g.readlines()
    with open("kinetic400.txt", 'r') as f:
        videos = f.readlines()
else:
    with open("\ucf101_val_split_1_videos.txt", 'r') as g:
        video_file_val = g.readlines()
    with open("ucf101.txt", 'r') as f:
        videos = f.readlines()
if args.models == "MVIT":
    channel = 8
    width = 112
elif args.models == "SLOWFAST":
    channel = 16
    width = 128
else:
    channel = 10
    width = 112
try:
    totalTime = time.time()
    success_number = 0
    number = 0
    for i_batch, file in enumerate(videos):
        video_name = file.split(" ")[0]
        true_label = int(file.split(" ")[-1].replace("\n", ''))
        print(osp.join(args.inputpath, video_name))
        data = load_attack_video(model, osp.join(args.inputpath, video_name))
        if args.models == "SLOWFAST":
            data['inputs'][0] = data['inputs'][0][0].unsqueeze(0)
            cover = ((data['inputs'][0]) / 255.0).to(device)
        elif args.models == "TSN":
            cover = ((data['inputs'][0]) / 255.0).permute(1,0,2,3).unsqueeze(0).to(device)
        else:
            cover = ((data['inputs'][0]) / 255.0).to(device)
        result = model.test_step(data)[0]
        labels = torch.min(result.pred_score.data, 0)[1].to(device)
        print(file, true_label, labels)
        for jk in video_file_val:
            if int(jk.split(" ")[1].replace("\n", '')) == labels.item():
                print(osp.join(args.inputpath, jk.split(" ")[0]))
                X_2 = load_attack_video(model, osp.join(args.inputpath, jk.split(" ")[0]))
                result_2 = model.test_step(X_2)[0]
                labels_2 = torch.max(result_2.pred_score.data, 0)[1].to(device)
                if labels_2.item() == labels.item():
                    if args.models == "TSN":
                        X_tgt = (X_2['inputs'][0] / 255.0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
                        break
                    elif args.models == "MVIT":
                        X_tgt = (X_2['inputs'][0] / 255.0).to(device)
                        break
                    else:
                        X_tgt = (X_2['inputs'][0][0] / 255.0).to(device)
                        break
        X_1 = torch.full(cover.shape, 0.00000000001).to(device)
        # 均匀输入
        X_ori = X_1.to(device)
        X_ori = Variable(X_ori, requires_grad=True)
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)
        if not os.path.exists(os.path.join(args.outputpath, 'clean_video')):
            clean_path = os.path.join(args.outputpath, "clean_video")
            os.mkdir(clean_path)
        else:
            clean_path = os.path.join(args.outputpath, "clean_video")
        np.save(osp.join(clean_path, "{}-ori".format(true_label)), data['inputs'][0])
        cover_dwt_1 = dwt3(cover, "haar")
        cover_low_0 = cover_dwt_1[0][0]
        cover_dwt_1 = torch.flatten(cover_dwt_1, start_dim=1, end_dim=2)
        for i_epoch in range(c.epochs):
            #################
            #    train:可逆网络的训练过程   #
            #################
            CGT = X_ori.to(device)
            CGT_dwt_1 = dwt3(CGT, "haar")
            CGT_dwt_1 = torch.flatten(CGT_dwt_1, start_dim=1, end_dim=2)
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)
            output_dwt_1 = INN_net(input_dwt_1).to(device)
            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 24).to(device)
            output_r_dwt_1 = output_dwt_1.narrow(1, 24, 24).to(device)
            output_steg_dwt_2 = output_steg_dwt_2.view(1, 8, 3, channel, width, width)
            output_steg_low_0 = output_steg_dwt_2[0][0]
            output_r_dwt_1 = output_r_dwt_1.view(1, 8, 3, channel, width, width)
            output_steg_1 = idwt3(output_steg_dwt_2, 'haar')
            output_r = idwt3(output_r_dwt_1, 'haar')
            output_steg_1 = torch.clamp(output_steg_1, min=0, max=1).to(device)
            eta = torch.clamp(output_steg_1 - cover, min=-c.eps, max=c.eps)
            output_steg_1 = torch.clamp(cover + eta, min=0, max=1)
            input_model = (output_steg_1 * 255.0)
            if args.models == "TSN":
                data['inputs'][0] = input_model.squeeze(0).permute(1, 0, 2, 3)
            else:
                data['inputs'][0] = input_model
            result = model.test_step(data)[0]
            out = result.pred_score
            adv_cost = nn.CrossEntropyLoss().to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            adv_loss = c.lambda_a * adv_cost(out, labels).to(device)
            loss_2 = c.beta_a * torch.mean(
                torch.mean(torch.sum(torch.sqrt(torch.sum(torch.pow(eta / 255.0, 2), dim=3)), dim=3), dim=2), dim=1)
            ll_loss = c.gama_a * MSE(cover_low_0, output_steg_low_0)
            total_loss = adv_loss + loss_2 + ll_loss
            optim1.zero_grad()
            optim2.zero_grad()
            total_loss.backward()
            optim1.step()

            CGT_model = (CGT * 255.0)
            if args.models == "TSN":
                data['inputs'][0] = CGT_model.squeeze(0).permute(1, 0, 2, 3)
            else:
                data['inputs'][0] = CGT_model
            C_result = model.test_step(data)[0]
            C_out = C_result.pred_score
            C_adv_loss = adv_cost(C_out, labels).to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            loss_mse = MSE(CGT, X_tgt).to(device)
            total_loss_tgt = c.lambda_b * loss_mse + c.beta_b * C_adv_loss
            total_loss_tgt.backward()
            optim2.step()

            weight_scheduler.step()
            lr_min = c.lr_min
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < lr_min:
                optim1.param_groups[0]['lr'] = lr_min
            print(
                "epoch = {}; adv_loss = {}; loss_2 ={}; ll_loss={}; total_loss = {};pre_label = {}/{};tgt_label = {}/{}.".format(
                    i_epoch, adv_loss,loss_2.item(),ll_loss.item(), total_loss.item(), torch.max(out, 0)[1].item(),
                    torch.max(result.pred_score, 0)[0].item(), torch.max(C_out, 0)[1].item(),
                    torch.max(C_result.pred_score, 0)[0].item()))
            adv_path = os.path.join(args.outputpath, "adv_video")
            tgt_path = os.path.join(args.outputpath, "tgt_video")
            r_path = os.path.join(args.outputpath, "r_video")
            eta_path = os.path.join(args.outputpath, "eta_video")
            if not os.path.exists(os.path.join(args.outputpath, "adv_video")):
                os.mkdir(adv_path)
                os.mkdir(tgt_path)
                os.mkdir(r_path)
                os.mkdir(eta_path)
            if torch.max(out, 0)[1].item() == labels and torch.max(result.pred_score, 0)[0].item() > 0.915:
                np.save(osp.join(adv_path,
                                 "{}-{}-adv".format(true_label, labels)), input_model.cpu().detach().numpy())
                np.save(osp.join(tgt_path,
                                 "{}-{}-tgt".format(true_label, torch.max(C_out, 0)[1].item())),
                        CGT_model.cpu().detach().numpy())
                np.save(osp.join(r_path, "{}-r".format(true_label)),
                        (output_r * 255.0).cpu().detach().numpy())
                np.save(osp.join(eta_path, "{}-eta".format(true_label)),
                        (eta * 255.0).cpu().detach().numpy())
                success_number = success_number + 1
                break
            else:
                if i_epoch == c.epochs - 1:
                    np.save(
                        osp.join(adv_path,
                                 "{}-{}-{}adv".format(true_label, labels, torch.max(out, 0)[1].item())),
                        input_model.cpu().detach().numpy())
                    np.save(osp.join(tgt_path,
                                     "{}-tgt".format(torch.max(C_out, 0)[1].item())), CGT_model.cpu().detach().numpy())
                    np.save(
                        osp.join(r_path, "{}-r".format(true_label)),
                        (output_r * 255.0).cpu().detach().numpy())
                    np.save(osp.join(eta_path, "{}-eta".format(true_label)),
                            (eta * 255.0).cpu().detach().numpy())
                    break
    print("success_number = ", success_number)
    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    print("Total cost time :" + str(time_cost))
except:
    print("Run error!")
    raise
