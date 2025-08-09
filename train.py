import argparse
import os
import time
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from src.dataset.ufgvc import UFGVCDataset
from src.augmentations.pkb import PatchKeepBlur, PatchCutout, FullImageBlur
from src.utils.metrics import topk_accuracy, macro_f1, AverageMeter

def build_transforms(args):
    resize_side = args.resize_side
    train_crop = args.train_crop
    pad_resize = transforms.Resize((resize_side, resize_side))
    aug_list = [pad_resize]
    if args.color_jitter:
        aug_list.append(transforms.ColorJitter(0.2,0.2,0.2,0.1))
    if args.hflip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if args.rotate:
        aug_list.append(transforms.RandomRotation(10))
    aug_list.append(transforms.RandomCrop(train_crop))
    if args.augmentation == 'pkb':
        aug_list.append(PatchKeepBlur(n=args.pkb_n, a_fraction=args.pkb_a_frac, sigma=args.pkb_sigma, placement=args.pkb_placement, seed=args.seed))
    elif args.augmentation == 'cutout':
        aug_list.append(PatchCutout(n=args.pkb_n, a_fraction=args.pkb_a_frac, placement=args.pkb_placement, seed=args.seed))
    elif args.augmentation == 'fullblur':
        aug_list.append(FullImageBlur(sigma=args.pkb_sigma))
    aug_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    train_transform = transforms.Compose(aug_list)
    val_transform = transforms.Compose([
        pad_resize,
        transforms.CenterCrop(train_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transform, val_transform

def build_model(args, num_classes:int):
    if args.model == 'resnet50':
        return timm.create_model('resnet50', pretrained=args.pretrained, num_classes=num_classes)
    elif args.model == 'vit':
        return timm.create_model('vit_small_patch16_384', pretrained=args.pretrained, img_size=args.train_crop, num_classes=num_classes)
    else:
        raise ValueError('Unsupported model')

def build_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError('Unsupported optimizer')

def adjust_lr(optimizer, epoch, args):
    if args.lr_decay_epochs and epoch in args.lr_decay_epochs:
        for g in optimizer.param_groups:
            g['lr'] *= args.lr_decay_gamma

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward(); optimizer.step()
        acc1, acc5 = topk_accuracy(outputs, targets, topk=(1,5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1, images.size(0))
        top5_meter.update(acc5, images.size(0))
    return loss_meter.avg, top1_meter.avg, top5_meter.avg

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = topk_accuracy(outputs, targets, topk=(1,5))
            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1, images.size(0))
            top5_meter.update(acc5, images.size(0))
            all_outputs.append(outputs.cpu()); all_targets.append(targets.cpu())
    all_outputs = torch.cat(all_outputs); all_targets = torch.cat(all_targets)
    f1 = macro_f1(all_outputs, all_targets)
    return loss_meter.avg, top1_meter.avg, top5_meter.avg, f1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cotton80')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--model', choices=['resnet50','vit'], default='resnet50')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--opt', choices=['sgd','adamw'], default='sgd')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lr-decay-epochs', type=int, nargs='*', default=[60,120])
    p.add_argument('--lr-decay-gamma', type=float, default=0.1)
    p.add_argument('--resize-side', type=int, default=440)
    p.add_argument('--train-crop', type=int, default=384)
    p.add_argument('--color-jitter', action='store_true')
    p.add_argument('--hflip', action='store_true')
    p.add_argument('--rotate', action='store_true')
    p.add_argument('--augmentation', choices=['none','pkb','cutout','fullblur'], default='none')
    p.add_argument('--pkb-n', type=int, default=4)
    p.add_argument('--pkb-a-frac', type=float, default=0.25)
    p.add_argument('--pkb-sigma', type=float, default=2.0)
    p.add_argument('--pkb-placement', choices=['random','dispersed','contiguous'], default='random')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--val-split', default='test')
    p.add_argument('--output', default='./outputs')
    p.add_argument('--save-best', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)
    train_tf, val_tf = build_transforms(args)
    train_set = UFGVCDataset(dataset_name=args.dataset, root=args.data_root, split='train', transform=train_tf)
    val_set = UFGVCDataset(dataset_name=args.dataset, root=args.data_root, split=args.val_split, transform=val_tf)
    num_classes = len(train_set.classes)
    model = build_model(args, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    best_top1 = 0.0; history = []
    for epoch in range(1, args.epochs+1):
        adjust_lr(optimizer, epoch-1, args)
        start = time.time()
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, loader_train, criterion, optimizer, device)
        val_loss, val_top1, val_top5, val_f1 = evaluate(model, loader_val, criterion, device)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}/{args.epochs} | LR {current_lr:.2e} | Train Loss {tr_loss:.3f} T1 {tr_top1:.3f} | Val Loss {val_loss:.3f} T1 {val_top1:.3f} T5 {val_top5:.3f} F1 {val_f1:.3f} | {elapsed:.1f}s')
        history.append({'epoch':epoch,'train_loss':tr_loss,'train_top1':tr_top1,'val_loss':val_loss,'val_top1':val_top1,'val_top5':val_top5,'val_f1':val_f1})
        # save best checkpoint inside loop
        if args.save_best and val_top1 > best_top1:
            best_top1 = val_top1
            save_path = Path(args.output)/f'best_{args.model}_{args.dataset}.pth'
            torch.save({'model':model.state_dict(), 'args':vars(args), 'best_top1':best_top1, 'epoch':epoch}, save_path)
            print(f'  New best top1 {best_top1:.3f} -> saved to {save_path}')
    history_path = Path(args.output)/f'history_{args.model}_{args.dataset}.json'
    with open(history_path,'w',encoding='utf-8') as f: json.dump(history,f,indent=2)
    print(f'History written to {history_path}')

if __name__ == '__main__':
    main()
