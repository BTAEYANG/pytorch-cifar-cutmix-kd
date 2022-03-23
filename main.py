"""Train CIFAR10 with PyTorch."""
import csv
import glob
import numpy as np
import torch.backends.cudnn as cudnn
import os
import argparse
from torch.optim import Adam, AdamW, Adamax, SGD
from torch.utils.tensorboard import SummaryWriter
from kd.util_kd import make_criterion
from models import *
from utils.dataUtil import getDataLoader
from utils.tools import progress_bar, rand_bbox

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate 0.02')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--beta', default=1.0, type=float, help='hyper parameter beta')
parser.add_argument('--cut_mix_prob', default=0.3, type=float,
                    help='cut_mix probability cifar10 0.3 cifar100 0.5 imagenet 1.0, if value == 0 no use cut_mix')
parser.add_argument('--epochs', default=550, type=int, help='epochs 550')
parser.add_argument('--split_factor', default=0.2, type=int, help='split factor')
parser.add_argument('--seed', default=66, type=int, help='seed')
parser.add_argument('--model_name', default="PreActResNet50", type=str, help='model_name')
parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer name')
parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str, help='lr scheduler')
parser.add_argument('--kd', default=False, type=bool, help='using kd for student')
parser.add_argument('--alpha', default=0.9, type=float, help='using kd for student, the value of alpha for kd loss')
parser.add_argument('--T', default=4, type=int, help='using kd for student, the value of T for softmax')
parser.add_argument('--trial', type=int, default=1, help='trial id')
parser.add_argument('--data_set', type=str, default='CIFAR10', help='select data set')
parser.add_argument('--num_classes', type=int, default=10, help='net final num classes')
parser.add_argument('--teacher_model', default="PreActResNet50", type=str, help='teacher model name')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
train_loader, validation_loader, test_loader = getDataLoader(split_factor=args.split_factor, seed=args.seed, data_set=args.data_set)

# Model
print('==> Building model..')
net = None

if args.model_name == 'PreActResNet18':
    net = PreActResNet18(num_classes=args.num_classes)
elif args.model_name == 'PreActResNet50':
    net = PreActResNet50(num_classes=args.num_classes)
elif args.model_name == 'PreActResNet101':
    net = PreActResNet101(num_classes=args.num_classes)
elif args.model_name == 'ResNet18':
    net = ResNet18(num_classes=args.num_classes)
elif args.model_name == 'ResNet50':
    net = ResNet50(num_classes=args.num_classes)
elif args.model_name == 'VGG19':
    net = VGG('VGG19')
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint，ready to test')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    if args.model_name == 'PreActResNet18':
        if args.kd:
            checkpoint = torch.load(f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}_no_cut_mix_kd_{args.trial}.pth')
        else:
            checkpoint = torch.load(f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}_no_cut_mix_{args.trial}.pth')
    else:
        checkpoint = torch.load(f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    checkpoint_epoch = checkpoint['epoch']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    print(f'==> Loading Checkpoint，acc：{best_acc}%；checkpoint_epoch：{checkpoint_epoch}, model_name：{model_name}, optimizer：{optimizer}, scheduler：{scheduler}')

criterion = nn.CrossEntropyLoss()
# optimizer = Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
# optimizer = Adamax(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
# optimizer = Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
# optimizer = Adamax(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# optimizer = AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)


def del_file():
    fileNames = glob.glob(f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}.pth')
    for fileName in fileNames:
        os.remove(fileName)
        print(f"Remove pth file: {fileNames}")


# Training
def train(epoch):
    global batch_idx
    print(f'Epoch: {epoch + 1}')
    train_loss = 0
    correct = 0
    total = 0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cut_mix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = net(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * (correct / total), correct, total))

    return train_loss / (batch_idx + 1), 100. * (correct / total)


def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validation_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validation_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'scheduler': args.lr_scheduler
        }

        if not os.path.isdir(f'./checkpoint/{args.data_set}'):
            os.mkdir(f'./checkpoint/{args.data_set}')
        del_file()

        print('Saving pth file')

        if args.cut_mix_prob == 0:
            if args.kd:
                csv_name = f"./checkpoint/{args.data_set}/checkpoint_{args.model_name}_no_cut_mix_info_kd_{args.trial}.csv"
            else:
                csv_name = f"./checkpoint/{args.data_set}/checkpoint_{args.model_name}_no_cut_mix_info_{args.trial}.csv"
        else:
            csv_name = f"./checkpoint/{args.data_set}/checkpoint_{args.model_name}_cut_mix_info.csv"

        with open(csv_name, mode='a', newline='', encoding='utf8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # columns_name
            csv_writer.writerow(["model_name", "optimizer", "scheduler", "epoch", "acc"])
            csv_writer.writerow([args.model_name, args.optimizer, args.lr_scheduler, epoch + 1, acc])
            csv_file.close()

        if args.cut_mix_prob == 0:
            if args.kd:
                torch.save(state, f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}_no_cut_mix_kd_{args.trial}.pth')
            else:
                torch.save(state, f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}_no_cut_mix_{args.trial}.pth')
        else:
            torch.save(state, f'./checkpoint/{args.data_set}/{args.model_name}_{args.optimizer}_{args.lr_scheduler}_cut_mix.pth')

        best_acc = acc

    return val_loss / (batch_idx + 1), 100. * (correct / total)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return test_loss / (batch_idx + 1), 100. * (correct / total)


def train_kd(epoch, student_net, teacher_net):
    global batch_idx
    print(f'KD Training Epoch: {epoch + 1}')
    student_net.train()
    train_kd_loss = 0
    correct = 0
    total = 0

    criterion = make_criterion(args.alpha, args.T)

    for batch_idx, (inputs, hard_targets) in enumerate(train_loader):
        inputs, hard_targets = inputs.to(device), hard_targets.to(device)
        # KD compute loss
        y_s = student_net(inputs)
        y_t = teacher_net(inputs)
        loss = criterion(y_t, y_s, hard_targets)

        # loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_kd_loss += loss.item()
        _, predicted = y_s.max(1)
        total += hard_targets.size(0)
        correct += predicted.eq(hard_targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_kd_loss / (batch_idx + 1), 100. * (correct / total), correct, total))

    return train_kd_loss / (batch_idx + 1), 100. * (correct / total)


if args.cut_mix_prob == 0:
    if args.kd:
        writer = SummaryWriter(f'./logs/{args.data_set}/{args.model_name}_no_cut_mix_kd_{args.trial}') 
    else:
        writer = SummaryWriter(f'./logs/{args.data_set}/{args.model_name}_no_cut_mix')
else:
    writer = SummaryWriter(f'./logs/{args.data_set}/{args.model_name}')

test_avg_loss = 0.0
test_avg_acc = 0.0
for epoch in range(start_epoch, start_epoch + args.epochs):
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    writer.add_scalar('lr', current_lr, (epoch + 1))

    if args.resume:
        test_loss, test_acc = test(epoch)
        test_avg_loss += test_loss
        test_avg_acc += test_acc
        if epoch != 0 and (epoch + 1) % 10 == 0:
            print(f"test_avg_loss：{test_avg_loss / 10}, acc：{test_avg_acc / 10} %")
            test_avg_loss = 0.0
            test_avg_acc = 0.0
    else:
        if args.kd:
            student_net = net
            teacher_net = PreActResNet50()
            teacher_net = teacher_net.to(device)
            print('==> Resuming teacher from checkpoint，ready to KD Training')
            pth_name = f'./checkpoint/{args.data_set}/{args.teacher_model}_{args.optimizer}_{args.lr_scheduler}_cut_mix.pth'
            checkpoint = torch.load(pth_name)
            teacher_net.load_state_dict(checkpoint['net'], strict=False)
            train_loss, train_acc = train_kd(epoch, student_net, teacher_net)
            val_loss, val_acc = val(epoch)
        else:
            train_loss, train_acc = train(epoch)
            val_loss, val_acc = val(epoch)
        # log epoch loss
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, (epoch + 1))
        # log epoch top1_acc
        writer.add_scalars("top1_acc", {"train": train_acc, "val": val_acc}, (epoch + 1))

    scheduler.step()

writer.close()
