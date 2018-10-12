import os
#import tqdm
import torch
import time
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import *
from lib.utils import *
from lib.loss import *
from lib.dataset import *
from collections import OrderedDict
import model.model_v4 as model_v4
from model.handwriteRes import *
from lib.augment import *
from lib.visualize import *
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from collections import Counter
from lib.config import Option

def train(train_loader, model, criterion, optimizer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        image_var = torch.tensor(images).cuda(async=True)
        label = torch.tensor(target).cuda(async=True)


        y_pred = model(image_var)

        loss = criterion(y_pred, label)
        
        losses.update(loss.item(), images.size(0))

        prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
        acc.update(prec, PRED_COUNT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))
    vis.plot('train_loss', losses.avg)
    vis.plot('train_acc', acc.avg)


def validate(val_loader, model, criterion, opt, best_precision, lowest_loss):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        image_var = torch.tensor(images).cuda(async=True)
        target = torch.tensor(labels).cuda(async=True)

        with torch.no_grad():
            y_pred = model(image_var)
            loss = criterion(y_pred, target)

        # measure accuracy and record loss
        prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        acc.update(prec, PRED_COUNT)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
    vis.plot('val_loss', losses.avg)
    vis.plot('val_acc', acc.avg)
    print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
          ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
    return acc.avg, losses.avg


def test(test_loader, model, opt):
    csv_map = OrderedDict({'filename': [], 'probability': []})
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            y_pred = model(image_var)

            smax = nn.Softmax(1)
            smax_out = smax(y_pred)

        csv_map['filename'].extend(filepath)
        for output in smax_out:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability'].append(prob)

    result = pd.DataFrame(csv_map)
    result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

    sub_filename, sub_label = [], []
    for index, row in result.iterrows():
        sub_filename.append(row['filename'])
        pred_label = np.argmax(row['probability'])
        if pred_label == 0:
            sub_label.append('norm')
        else:
            sub_label.append('defect%d' % pred_label)

    return {'filename': sub_filename, 'label': sub_label}

#    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
#    submission.to_csv('./result/%s/submission.csv' % opt.file_name, header=None, index=False)
#    return

def filter_results(results):
    # get the total number of testdataset
    numbers = len(results[0]['filename'])
    
    sub_filename, sub_label = [], []
    
    for item in range(numbers):
    
        single_filename = list()
        single_result = list()
    
        for model in range(len(results)):
            single_filename.append(results[model]['filename'][item])
            single_result.append(results[model]['label'][item])
        
        assert(len(set(single_filename)) == 1)
        sub_label.append(Counter(single_result).most_common(1)[0][0])
        sub_filename.append(single_filename[0])
    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    submission.to_csv('./result/%s/resnet_original_bachsize_CV_submission_0928.csv' % opt.file_name, header=None, index=False)
    return
        
        

if __name__ == '__main__':

    # 随机种子
    np.random.seed(666)
    torch.manual_seed(24)
    torch.cuda.manual_seed_all(2018)
    random.seed(3421)

    opt = Option()
    vis = Visualizer(env='Tianchi')

    if not os.path.exists('./model/%s' % opt.file_name):
        os.makedirs('./model/%s' %opt.file_name)
    if not os.path.exists('./result/%s' % opt.file_name):
        os.makedirs('./result/%s' % opt.file_name)
    
    if not os.path.exists('./result/%s.txt' % opt.file_name):
        with open('./result/%s.txt' % opt.file_name, 'w') as acc_file:
            pass
    with open('./result/%s.txt' % opt.file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), opt.file_name))

    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'


    def save_checkpoint(state, is_best, is_lowest_loss, cv_num):
        if is_best:
            filename='./model/{}/model_best_{}.pkl'.format(opt.file_name, cv_num)
            torch.save(state['state_dict'], filename)
            

    def adjust_learning_rate():
        
        lr = opt.lr / opt.lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=opt.weight_decay, amsgrad=True)



#    model = model_v4.v4(num_classes=12)
#    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        checkpoint_path = './model/%s/checkpoint.pth.tar' % opt.file_name
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])

            if start_epoch in np.cumsum(opt.stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % opt.file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    all_data = pd.read_csv('data/label.csv')
    
    test_data_list = pd.read_csv('data/test.csv')
    test_data = TestDataset(test_data_list, transform=testAugment)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size*2, shuffle=False, pin_memory=False, num_workers=opt.workers)
    
    
    # to try k_fold
    
    cv = StratifiedKFold(n_splits=opt.kfolds, shuffle=True, random_state=1337)
    
    for cv_num, (train_idx, val_idx) in enumerate(cv.split(all_data['img_path'], all_data['label'])):
        
        #model = model_v4.v4(num_classes=12)
        #model = myResNet152()
        model = get_resnet152()

#        resnet152 = models.resnet152(pretrained=True)
#         
#        pretrained_dict = resnet152.state_dict()  
#        
#        model_dict = model.state_dict()  
#         
#        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
#        
#        model_dict.update(pretrained_dict)  
#         
#        model.load_state_dict(model_dict) 
        
        model = torch.nn.DataParallel(model).cuda()
        
        
    
        criterion = nn.CrossEntropyLoss().cuda()
        #criterion = FocalLoss(class_num = 12).cuda()
    
        optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay, amsgrad=True)
        
        train_img_path = all_data['img_path'][train_idx]
        train_label =  all_data['label'][train_idx]
        train_data_list = pd.concat([train_img_path, train_label], axis = 1)
        
        val_img_path = all_data['img_path'][val_idx]
        val_label =  all_data['label'][val_idx]
        val_data_list = pd.concat([val_img_path, val_label], axis = 1)
        
        train_data = TrainDataset(train_data_list, transform=trainAugment)

        val_data = ValDataset(val_data_list, transform=testAugment)

        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.workers)
        
        val_loader = DataLoader(val_data, batch_size=opt.batch_size*2, shuffle=False, pin_memory=False, num_workers=opt.workers)
        
        # set the original loss and precision
        best_precision = 0
        lowest_loss = 100

        for epoch in range(opt.start_epoch, opt.total_epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, opt)
            
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion, opt, best_precision, lowest_loss)


            with open('./result/%s.txt' % opt.file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))


            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': opt.stage,
                'lr': opt.lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss, cv_num)


            if (epoch + 1) in np.cumsum(opt.stage_epochs)[:-1]:
                opt.stage += 1
                optimizer = adjust_learning_rate()
                model_name = './model/{}/model_best_{}.pkl'.format(opt.file_name, cv_num)
                if os.path.exists(model_name):
                    model.load_state_dict(torch.load(model_name))
                print('Step into next stage')
                with open('./result/%s.txt' % opt.file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')


        with open('./result/%s.txt' % opt.file_name, 'a') as acc_file:
            acc_file.write('* best acc: %.8f  %s  cv_num: %s\n' % (best_precision, os.path.basename(__file__), str(cv_num)))
        with open('./result/best_acc.txt', 'a') as acc_file:
            acc_file.write('%s  * best acc: %.8f  %s  cv_num: %s\n' % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision,
                os.path.basename(__file__),  str(cv_num)))

    results = list()
    for i in range(int(opt.kfolds)):
        model.load_state_dict(torch.load('./model/{}/model_best_{}.pkl'.format(opt.file_name, i)))
        results.append(test(test_loader=test_loader, model=model, opt=opt))

    filter_results(results)

    torch.cuda.empty_cache()























