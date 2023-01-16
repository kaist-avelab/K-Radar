'''
This demo is used to validate the back-propagation of the torch implementation of oriented
2d box intersection. 

This demo trains a network which takes N set of box corners and predicts the x, y, w, h and angle
of each rotated boxes. In order to do the back-prop, the prediected box parameters and GT are 
converted to coordinates of box corners. The area of intersection area is calculated using 
the pytorch function with CUDA extension. Then, the GIoU loss or DIoU loss can be calculated.

This demo first generates data and then do the training.

The network is simply a shared MLP (implemented as Conv-layers with 1x1 kernel).

Lanxiao Li
2020.08
'''

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import argparse
from utiles import box2corners
from oriented_iou_loss import cal_diou, cal_giou

DATA = "./data"
X_MAX = 3
Y_MAX = 3
SCALE = 0.5
BATCH_SIZE = 32
N_DATA = 128
NUM_TRAIN = 200 * BATCH_SIZE * N_DATA
NUM_TEST = 20 * BATCH_SIZE * N_DATA

NUM_EPOCH = 20

if not os.path.exists(DATA):
    os.mkdir(DATA)

def create_data(num):
    print("... generating %d boxes, please wait..."%num)
    x = (np.random.rand(num) - 0.5) * 2 * X_MAX
    y = (np.random.rand(num) - 0.5) * 2 * Y_MAX
    w = (np.random.rand(num) - 0.5) * 2 * SCALE + 1
    h = (np.random.rand(num) - 0.5) * 2 * SCALE + 1
    alpha = np.random.rand(num) * np.pi
    corners = np.zeros((num, 4, 2)).astype(np.float)
    for i in range(num):
        corners[i, ...] = box2corners(x[i], y[i], w[i], h[i], alpha[i])
    label = np.stack([x, y , w, h, alpha], axis=1)
    return corners, label

def save_dataset():
    train_data, train_label = create_data(NUM_TRAIN)
    np.save(os.path.join(DATA, "train_data.npy"), train_data)
    np.save(os.path.join(DATA, "train_label.npy"), train_label)
    test_data, test_label = create_data(NUM_TEST)
    np.save(os.path.join(DATA, "test_data.npy"), test_data)
    np.save(os.path.join(DATA, "test_label.npy"), test_label)
    print("data saved in: ", DATA)

class BoxDataSet(Dataset):
    def __init__(self, split="train"):
        super(BoxDataSet, self).__init__()
        assert split in ["train", "test"], "split must be train or test"
        self.split = split
        try:
            self.data = np.load(os.path.join(DATA, split+"_data.npy"))
            self.label = np.load(os.path.join(DATA, split+"_label.npy"))
        except:
            save_dataset()
            self.data = np.load(os.path.join(DATA, split+"_data.npy"))
            self.label = np.load(os.path.join(DATA, split+"_label.npy"))
    def __len__(self) -> int:
        return self.data.shape[0]
    def __getitem__(self, index: int) :
        d = self.data[index, ...]
        l = self.label[index, ...]
        return torch.FloatTensor(d), torch.FloatTensor(l)

def create_network():
    return nn.Sequential(nn.Conv1d(8, 128, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 512, 1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Conv1d(512, 128, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 5, 1),
                nn.Sigmoid())

def parse_pred(pred:torch.Tensor):
    p0 = (pred[..., 0] - 0.5) * 2 * X_MAX
    p1 = (pred[..., 1] - 0.5) * 2 * Y_MAX
    p2 = (pred[..., 2] - 0.5) * 2 * SCALE + 1
    p3 = (pred[..., 3] - 0.5) * 2 * SCALE + 1
    p4 = pred[..., 4] * np.pi
    return torch.stack([p0,p1,p2,p3,p4], dim=-1)

def main(loss_type:str="giou", enclosing_type:str="aligned"):
    ds_train = BoxDataSet("train")
    ds_test = BoxDataSet("test")
    ld_train = DataLoader(ds_train, BATCH_SIZE * N_DATA, shuffle=True, num_workers=4)
    ld_test = DataLoader(ds_test, BATCH_SIZE * N_DATA, shuffle=False, num_workers=4)
    
    net = create_network()
    net.to("cuda:0")
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_batch = len(ds_train)//(BATCH_SIZE*N_DATA)
    
    for epoch in range(1, NUM_EPOCH+1):
        # train
        net.train()
        for i, data in enumerate(ld_train, 1):
            box, label = data
            box = box.cuda()                            # (B*N, 4, 2)
            box = box.view([BATCH_SIZE, -1, 4*2])       # (B, N, 4*2)
            box = box.transpose(1, 2)                   # (B, 8, N)
            label = label.cuda()                        # (B*N, 5)
            label = label.view([BATCH_SIZE, -1, 5])     # (B, N, 5)
            
            optimizer.zero_grad()
            pred = net(box)                             # (B, 5, N)
            pred = pred.transpose(1,2)                  # (B, N, 5)
            pred = parse_pred(pred)

            iou_loss, iou = None, None
            if loss_type == "giou":
                iou_loss, iou = cal_giou(pred, label, enclosing_type)
            elif loss_type == "diou":
                iou_loss, iou = cal_diou(pred, label, enclosing_type)
            else:
                ValueError("unknown loss type")
            iou_loss = torch.mean(iou_loss)
            iou_loss.backward()
            optimizer.step()

            if i%10 == 0:
                iou_mask = (iou > 0).float()
                mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                print("[Epoch %d: %d/%d] train loss: %.4f  mean_iou: %.4f"
                    %(epoch, i, num_batch, iou_loss.detach().cpu().item(), mean_iou.detach().cpu().item()))
        lr_scheduler.step()

        # validate
        net.eval()
        aver_loss = 0
        aver_mean_iou = 0
        with torch.no_grad():
            for i, data in enumerate(ld_test, 1):
                box, label = data
                box = box.cuda()                            # (B*N, 4, 2)
                box = box.view([BATCH_SIZE, -1, 4*2])       # (B, N, 4*2)
                box = box.transpose(1, 2)                   # (B, 8, N)
                label = label.cuda()                        # (B*N, 5)
                label = label.view([BATCH_SIZE, -1, 5])     # (B, N, 5)
                
                pred = net(box)                             # (B, 5, N)
                pred = pred.transpose(1,2)                  # (B, N, 5)
                pred = parse_pred(pred)

                iou_loss, iou = None, None
                if loss_type == "giou":
                    iou_loss, iou = cal_giou(pred, label, enclosing_type)
                elif loss_type == "diou":
                    iou_loss, iou = cal_diou(pred, label, enclosing_type)
                else:
                    ValueError("unknown loss type")
                iou_loss = torch.mean(iou_loss)
                aver_loss += iou_loss.cpu().item()
                iou_mask = (iou > 0).float()
                mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                aver_mean_iou += mean_iou.cpu().item()
        print("... validate epoch %d ..."%epoch)
        n_iter = len(ds_test)/BATCH_SIZE/N_DATA
        print("average loss: %.4f"%(aver_loss/n_iter))
        print("average iou: %.4f"%(aver_mean_iou/n_iter))
        print("..............................")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="diou", help="type of loss function. support: diou or giou. [default: diou]")
    parser.add_argument("--enclosing", type=str, default="smallest", 
        help="type of enclosing box. support: aligned (axis-aligned) or pca (rotated) or smallest (rotated). [default: smallest]")
    flags = parser.parse_args()
    main(flags.loss, flags.enclosing)