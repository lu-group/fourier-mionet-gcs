import deepxde as dde
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 

def get_data(ntrain, ntest):
    z = np.linspace(0, 1, 96).astype(np.float32)
    t = np.linspace(0, 1, 24).astype(np.float32)
    r = np.linspace(0, 1, 200).astype(np.float32)
    xrt = np.array([[c] for c in t])
    xrt = xrt.astype(np.float32)

    field_input = [True,True,True,False,False,False,False,False,False,True,True]
    x_train = np.load("sg_train_a.npz")["sg_train_a"][:ntrain, :, :, field_input].transpose(0,1,2,3).astype(np.float32)
    x_train_MIO = np.load("sg_train_a_MIO.npy")[:ntrain,:].astype(np.float32)

    grid_x = np.load("sg_train_a.npz")["sg_train_a"][0,0,:,-2].astype(np.float32)
    

    x_train = (x_train,x_train_MIO, xrt)

    y_train = np.load("sg_train_u.npz")["sg_train_u"][:ntrain, :, :, :].transpose(0,3,1,2).reshape(ntrain, 24*96*200).astype(np.float32)


    
    x_test = np.load("sg_test_a.npz")["sg_test_a"][-ntest:, :, :, field_input].transpose(0,1,2,3).astype(np.float32)
    
    x_test_MIO = np.load("sg_test_a_MIO.npy")[-ntest:,:].astype(np.float32)

    x_test = (x_test,x_test_MIO, xrt)
    
    y_test = np.load("sg_test_u.npz")["sg_test_u"][-ntest:, :, :, :].transpose(0,3,1,2).reshape(ntest, 24*96*200).astype(np.float32)

    
    return x_train, y_train, x_test, y_test, grid_x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):

        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-2,-1])


        out_ft = torch.zeros(batchsize, self.out_channels,x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)
        

        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

class decoder(nn.Module):
    def __init__(self, modes1, modes2, width,width2):
        super(decoder, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)

        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)

        self.fc1 = nn.Linear(self.width, width2)
        self.fc2 = nn.Linear(width2, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y =  x.shape[2], x.shape[3]
        
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet3(x) 
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        x = x.view(batchsize, size_x, size_y, 1)[..., :-8,:-8, :]
        
        return x.squeeze()

class branch1(nn.Module):
    def __init__(self,width):
        super(branch1, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(5, self.width)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        x = F.pad(F.pad(x, (0,0,0,8), "replicate"), (0,0,0,0,0,8), 'constant', 0)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

class branch2(nn.Module):
    def __init__(self,width):
        super(branch2, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(7, self.width)


    def forward(self, x):
        x = self.fc0(x)
        
        return x

x_train, y_train, x_test, y_test, grid_x = get_data(4500, 500)


data = dde.data.QuadrupleCartesianProd(x_train, y_train, x_test, y_test)


scaler = StandardScaler().fit(y_train)
std = np.sqrt(scaler.var_.astype(np.float32))

def output_transform(inputs, outputs):
    
    return outputs * torch.as_tensor(std) + torch.as_tensor(scaler.mean_.astype(np.float32))
    

gelu = torch.nn.GELU()

Net = dde.nn.pytorch.mionet.MIONetCartesianProd(
 layer_sizes_branch1=[4500*96*200*3,branch1(36)],layer_sizes_branch2=[7*36,branch2(36)],layer_sizes_trunk=[1,36,36,36,36],
 activation={"branch1":gelu,
        "branch2":gelu,
        "trunk":gelu,
        "merger":gelu,
        "output merger":gelu},
        kernel_initializer="Glorot normal",
        regularization=("l2",4e-6),
        trunk_last_activation=False,
        merge_operation="sum",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=[36,decoder(10,10,36,128)])

z = np.linspace(0, 1, 96).astype(np.float32)
grid_dy = z[:-1]-z[1:]
grid_dy = grid_dy.reshape(95,1)
grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
grid_dx = torch.as_tensor(grid_dx).reshape(1,1,1,198)
pre_mask_train = np.isclose(y_train.reshape(y_train.shape[0],24,96,200)[:,-1,:,0],0.0)
pre_mask_test = np.isclose(y_test.reshape(500,24,96,200)[:,-1,:,0],0.0)
pre_mask_train = torch.as_tensor(pre_mask_train)
pre_mask_test = torch.as_tensor(pre_mask_test)
grid_dy = torch.as_tensor(grid_dy)

def loss_fnc(y_true, y_pred,train_indices,istrain):
    size = y_true.shape[0]
    timesize = int(y_true.shape[1]/200/96)
    y_true = y_true.reshape(size,timesize,96,200)
    y_pred = y_pred.reshape(size,timesize,96,200)
    if istrain:
        mask = pre_mask_train[train_indices]
    else:
        mask = pre_mask_test[train_indices]
    mask = 1-mask.to(torch.float32)
    mask = mask.reshape(size,1,96,1)  
    y_true = y_true*mask
    y_pred = y_pred*mask
    dydx_true_x = (y_true[:,:,:,2:]-y_true[:,:,:,:-2])/grid_dx
    dydx_pred_x = (y_pred[:,:,:,2:]-y_pred[:,:,:,:-2])/grid_dx
    y_true = y_true.reshape(size,timesize*96*200)
    y_pred = y_pred.reshape(size,timesize*96*200)
    dydx_true_x = dydx_true_x.reshape(size,timesize*96*198)
    dydx_pred_x = dydx_pred_x.reshape(size,timesize*96*198)
    ori_loss =  torch.mean(torch.norm(y_true - y_pred, 2, dim=1) / torch.norm(y_true, 2, dim=1))
    der_loss_x =  torch.mean(torch.norm(dydx_true_x - dydx_pred_x, 2, dim=1) / torch.norm(dydx_true_x, 2, dim=1))

    return [ori_loss, der_loss_x]

def Rsquare_plume(y_true,y_pred):
    size = y_true.shape[0]
    y_true = y_true.reshape(size,24,96,200)
    y_pred = y_pred.reshape(size,24,96,200)
    sse = 0
    sst = 0
    r2 = 0
    for i in range(size):
        z_axis = y_true[i,-1,:,0]
        mask = np.isclose(z_axis,0.0)
        mask=1-mask
        mask=mask.astype(bool)
        for j in range(24):
            
            y_true_i = y_true[i,j,mask,:]
            y_pred_i = y_pred[i,j,mask,:]
            sse = np.sum(np.square(y_true_i.flatten()-y_pred_i.flatten()))
            sst = np.sum(np.square(y_true_i.flatten()-np.mean(y_true_i.flatten())))
            r2 += 1-sse/sst
    return r2/24/size
    
def Rsquare_plume_tegother(y_true,y_pred):
    size = y_true.shape[0]
    y_true = y_true.reshape(size,24,96,200)
    y_pred = y_pred.reshape(size,24,96,200)
    sse = 0
    sst = 0
    r2 = 0
    for i in range(size):
        z_axis = y_true[i,-1,:,0]
        mask = np.isclose(z_axis,0.0)
        mask=1-mask
        mask=mask.astype(bool)
        y_true_i = y_true[i][:,mask,:]
        y_pred_i = y_pred[i][:,mask,:]
        sse = np.sum(np.square(y_true_i.flatten()-y_pred_i.flatten()))
        sst = np.sum(np.square(y_true_i.flatten()-np.mean(y_true_i.flatten())))
        r2 += 1-sse/sst
    return r2/size

path = f'./pre_train/'
if not os.path.exists(path):
    os.mkdir(path)

model = dde.Model(data, Net)
model.compile("adam", loss=loss_fnc,loss_weights=[1,0.5],
                  lr=1e-3, decay=("step", 3375, 0.9), metrics=[Rsquare_plume_tegother,Rsquare_plume] )
checker = dde.callbacks.ModelCheckpoint("pre_train/sg_model",save_better_only=True,period=3375,monitor="test loss")
losshistory, train_state = model.train(epochs=168750, batch_size=4,timestep_batch_size=8,training_time_size=24,display_every=3375,callbacks=[checker])


print(model.net.num_trainable_parameters())

