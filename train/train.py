import sys, torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
import data_loader as ml
from model.cnn import Net


# import pickle

# LOSS = []

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--use_cuda', default=True, help='using CUDA for training')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True


def train():
    total_step = 0
    writer = SummaryWriter("logs")
    os.makedirs('./output', exist_ok=True)
    # if True: #not os.path.exists('output/total.txt'):
    #     ml.image_list(args.datapath, 'output/total.txt')
    #     ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

    train_data = ml.MyDataset(root=r'E:\Dataset\DIV2K_train_HR\DIV2K_train_HR', file="train.txt", width=512, height=512)
    val_data = ml.MyDataset(root=r'E:\Dataset\DIV2K_train_HR\DIV2K_train_HR', file="val.txt", width=512, height=512)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    model = Net()
    # model = models.resnet18(num_classes=10)  # 调用内置模型
    # model.load_state_dict(torch.load('./output/params_10.pth'))
    # from torchsummary import summary
    # summary(model, (3, 28, 28))

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    # loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    loss_func = nn.MSELoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()

            """
            PIL_img_input = transforms.ToPILImage()(batch_x[0])
            PIL_img_output = transforms.ToPILImage()(out[0])
            PIL_img_label = transforms.ToPILImage()(batch_y[0])
            PIL_img_input.show()
            PIL_img_output.show()
            PIL_img_label.show()
            """

            # pred = torch.max(out, 1)[1]

            # train_correct = (pred == batch_y).sum() / (256 * 256)
            """
            train_correct = (out == batch_y).sum() / (3 * 256 * 256)
            train_acc += train_correct.item()
            """
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item()))

            # global LOSS
            # LOSS.append(loss.item())
            # pickle.dump(LOSS, open("./buffer/buffer.pkl", "wb"))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss.item(), total_step)
            total_step += 1

        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f' % (train_loss / (math.ceil(len(train_data) / args.batch_size))))
        writer.add_scalar("epoch_loss", train_loss / (math.ceil(len(train_data) / args.batch_size)), epoch)

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            # pred = torch.max(out, 1)[1]
            # num_correct = (pred == batch_y).sum() / (256 * 256)
            """
            num_correct = (out == batch_y).sum() / (3 * 256 * 256)
            eval_acc += num_correct.item()
            """
        print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(val_data) / args.batch_size))))
        writer.add_scalar("test_loss", eval_loss / (math.ceil(len(val_data) / args.batch_size)), epoch)

        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')
            # to_onnx(model, 3, 28, 28, 'params.onnx')
    writer.close()


if __name__ == '__main__':
    train()

