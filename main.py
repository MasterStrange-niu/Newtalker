import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from tensorboardX import SummaryWriter
from data_loader import get_dataloaders
from newtalker import Newtalker
#from testvocaset import cs
def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=1000):
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    #writer = SummaryWriter(log_dir='./runs/exp_name')
    writer = SummaryWriter('./tensorboard')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0

    for e in range(epoch+1):
        loss_log = []
        loss_log2 = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()
        result_path = os.path.join(args.dataset, args.result_path)
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)
        result_path2 = os.path.join(args.dataset, args.result_path2)
        if os.path.exists(result_path2):
            shutil.rmtree(result_path2)
        os.makedirs(result_path2)

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss,loss2,lvp ,vertice_out,d= model(audio, template,  vertice, criterion,teacher_forcing=False)
            #loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            loss_log2.append(loss2.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.10f} TRAIN LOSS:{:.10f}".format((e+1), iteration ,np.mean(loss_log),np.mean(loss_log2)))
            #pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.10f} ".format((e+1), iteration ,np.mean(loss_log)))
        # validation
        valid_loss_log = []
        valid_loss_log2 = []
        model.eval()
        vertices_gt_all = []
        vertices_pred_all = []
        motion_std_difference = []
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            loss, loss2, lvp, vertice_out,d = model(audio, template, vertice, criterion)
            valid_loss_log.append(loss.item())
            valid_loss_log2.append(loss2.item())

            
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+".npy") , vertice_out.detach().cpu().numpy())
            np.save(os.path.join(result_path2, file_name[0].split(".")[0]+".npy") , d.detach().cpu().numpy())
            #exit()
        if args.dataset == "vocaset":
            lve,fdd=cs()

        if args.dataset == "BIWI":
            lve,fdd=cs2()
            print (lve,fdd)
        #if lve<3.03e-05 :#and fdd<4.5e-07:
        if lve<4.03e-04 and fdd<3.3e-05:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

       

        current_loss = np.mean(valid_loss_log)
        current_loss2 = np.mean(valid_loss_log2)
        writer.add_scalar('lve', lve, global_step=e)
        writer.add_scalar('fdd', fdd, global_step=e)
        writer.add_scalar('lve/fdd', lve, global_step=fdd)
        #if (e > 0 and e % 1000 == 0) or e == args.max_epoch:
        ###    torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.10f} ,current loss2:{:.10f}".format(e+1,current_loss,current_loss2))
    return model

@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.dataset, args.result_path)
    print(result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(
            device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(
                    os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def cs(train_sub="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA",
       pred_path="./vocaset/result/",
       gt_path="./vocaset/result2/",
       region_path="./vocaset/regions/",
       templates_path="./vocaset/templates.pkl"):  
    train_subject_list = train_sub.split(" ")
    sentence_list = ["sentence"+str(i).zfill(2) for i in range(21,41) if i != 32]
    with open(templates_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    with open(os.path.join(region_path, "vocaset-lve-new.txt")) as f:
        maps = f.read().split(",")
        mouth_map = [int(i) for i in maps if i != '']
    with open(os.path.join(region_path, "vocaset-fdd-new.txt")) as f:
        maps = f.read().split(",")
        upper_map = [int(i) for i in maps if i != '']    
    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []
    for subject in train_subject_list:
        for sentence in sentence_list:
            vertices_gt = np.load(os.path.join(gt_path,subject+"_"+sentence+".npy")).reshape(-1,5023,3)
            vertices_pred = np.load(os.path.join(pred_path,subject+"_"+sentence+".npy")).reshape(-1,5023,3)
            vertices_pred = vertices_pred[:vertices_gt.shape[0],:,:]
            motion_pred = vertices_pred - templates[subject].reshape(1,5023,3)
            motion_gt = vertices_gt - templates[subject].reshape(1,5023,3)
            cnt += vertices_gt.shape[0]
            vertices_gt_all.extend(list(vertices_gt))
            vertices_pred_all.extend(list(vertices_pred))
            L2_dis_upper = np.array([np.square(motion_gt[:,v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
            L2_dis_upper = np.sum(L2_dis_upper,axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)           
            L2_dis_upper = np.array([np.square(motion_pred[:,v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
            L2_dis_upper = np.sum(L2_dis_upper,axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)
            motion_std_difference.append(gt_motion_std - pred_motion_std)
    print('Frame Number: {}'.format(cnt))
    vertices_gt_all = np.array(vertices_gt_all)
    print(vertices_gt_all.shape)
    vertices_pred_all = np.array(vertices_pred_all)
    print(vertices_pred_all.shape)
    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1,0,2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max,axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max,axis=1)
    print('Lip Vertex Error: {:.4e}'.format(np.mean(L2_dis_mouth_max)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference)/len(motion_std_difference)))
    return np.mean(L2_dis_mouth_max), sum(motion_std_difference)/len(motion_std_difference)
def cs2(train_subjects="F2 F3 F4 M3 M4 M5",
        pred_path="./BIWI/result/",
        gt_path="./BIWI/result2/",
        region_path="./BIWI/regions/",
        templates_path="./BIWI/templates.pkl"):
    train_subject_list = train_subjects.split(" ")
    sentence_list = ["e"+str(i).zfill(2) for i in range(37,41)]
    with open(templates_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    with open(os.path.join(region_path, "lve.txt")) as f:
        maps = f.read().split(", ")
        mouth_map = [int(i) for i in maps]
    with open(os.path.join(region_path, "fdd.txt")) as f:
        maps = f.read().split(", ")
        upper_map = [int(i) for i in maps]
    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []
    for subject in train_subject_list:
        for sentence in sentence_list:
            vertices_gt = np.load(os.path.join(gt_path,subject+"_"+sentence+".npy")).reshape(-1,23370,3)
            vertices_pred = np.load(os.path.join(pred_path,subject+"_"+sentence+".npy")).reshape(-1,23370,3)
            vertices_pred = vertices_pred[:vertices_gt.shape[0],:,:]
            motion_pred = vertices_pred - templates[subject].reshape(1,23370,3)
            motion_gt = vertices_gt - templates[subject].reshape(1,23370,3)
            cnt += vertices_gt.shape[0]
            vertices_gt_all.extend(list(vertices_gt))
            vertices_pred_all.extend(list(vertices_pred))
            L2_dis_upper = np.array([np.square(motion_gt[:,v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
            L2_dis_upper = np.sum(L2_dis_upper,axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)
            L2_dis_upper = np.array([np.square(motion_pred[:,v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
            L2_dis_upper = np.sum(L2_dis_upper,axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)
            motion_std_difference.append(gt_motion_std - pred_motion_std)
    print('Frame Number: {}'.format(cnt))
    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)    
    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1,0,2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max,axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max,axis=1)
    lip_vertex_error = np.mean(L2_dis_mouth_max)
    fdd = sum(motion_std_difference)/len(motion_std_difference)
    print('Lip Vertex Error: {:.4e}'.format(lip_vertex_error))
    print('FDD: {:.4e}'.format(fdd))
    return lip_vertex_error, fdd   
def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save2", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--result_path2", type=str, default="result2", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")

    args = parser.parse_args()

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    # Load previously trained weights if available
    checkpoint_path = os.path.join(args.dataset, args.save_path, '_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded checkpoint from epoch 150")

    
    model = trainer(args, dataset["train"], dataset["test"],model, optimizer, criterion, epoch=args.max_epoch)
    writer.close()
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()
