import torch
from torchvision.datasets.caltech import Caltech101
from TuneModel import TuneNet
from torchvision.transforms import transforms
from torch import nn
from tqdm import tqdm
from torch import optim
import numpy as np
from dataset import TuneDataSet
from torch.utils.data.dataloader import DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, model, batch_size, epoch, optimizer, criterion1,criterion2,device=None):
    model.train()

    train_loss = 0
    step_length = int(train_loader.len/batch_size)
    pbr = tqdm(range(step_length+1))

    for batch_idx, (x1,x2,target) in zip(pbr, train_loader):

        x1,x2, target = x1.to(device), x2.to(device),target.to(device)
        optimizer.zero_grad()
        out_param,out_img = model(x1,x2)
        loss1 = criterion1(out_param,target)
        loss2 = criterion2(out_img, x2)

        loss = loss1+loss2

        train_loss += loss.__float__()

        optimizer.second_step(zero_grad=True)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        # optimizer.step()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if batch_idx % 10 == 0:
            pbr.set_description('Train Epoch: {} Loss: {:.6f} lr:{:.7f}'.format(
                epoch, loss.item(), lr))

    train_loss /= step_length
    return train_loss


def val(val_loader, model, criterion1,criterion2, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        cnt = 0
        for batch_idx, (x1,x2,target) in enumerate(val_loader):
            cnt += 1
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            optimizer.zero_grad()
            out_param, out_img = model(x1, x2)
            loss1 = criterion1(out_param, target)
            loss2 = criterion2(out_img, x2)

            loss = loss1 + loss2
            test_loss += loss.__float__()

        test_loss /= cnt

        print('Val: Average loss: {:.4f}'.format(test_loss)

        return test_loss




def get_dataloader(root, batch_size):
    train_dataset = TuneDataSet(root, train_transform, is_train=True)
    val_dataset = TuneDataSet(root, val_transform,is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader,val_loader

if __name__ == '__main__':
    batch_size = 16
    total_epoch = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = "./checkpoints/TuneNet/"


    model = TuneNet()

    # define transforms
    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])


    train_loader,val_loader = get_dataloader()

    criterion1 = nn.L1Loss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)

    train_loss = []
    test_loss = []
    min_loss = 1000
    min_epoch = 0
    best_model = None
    # 开始训练
    for epoch in range(1, total_epoch + 1):
        # ----------------------------训练----------------------------
        tr_loss, tr_acc = train(train_loader, model,batch_size, epoch, optimizer, criterion1,criterion2, device=device)
        # 记录训练loss和acc
        train_loss.append(tr_loss)

        # --------------------------验证--------------------------------
        ts_loss = val(val_loader, model, criterion1,criterion2, device)
        # 记录验证的loss和acc
        test_loss.append(ts_loss)

        # 保存最好的模型
        if ts_loss < min_loss:
            min_loss = ts_loss
            min_epoch = epoch
            best_model = model
            # 保存最好的模型和参数
            torch.save(model, save_folder+'best.pkl')

        # 更新学习率
        scheduler.step()
