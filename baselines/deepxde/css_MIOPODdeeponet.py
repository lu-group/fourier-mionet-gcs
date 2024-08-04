import deepxde as dde
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as func
from sklearn.decomposition import PCA

# def get_data(ntrain, ntest):
#     z = np.linspace(0, 1, 96).astype(np.float32)
#     r = np.linspace(0, 1, 200).astype(np.float32)
#     xrt = np.array([[a, b] for a in z for b in r])
    
#     x_train_branch = np.load("dP_train_a.npz")["dP_train_a"][:ntrain, :, :, :-2].transpose(0,3,1,2).astype(np.float32)
#     x_train = (x_train_branch, xrt)
    
#     y_train = np.load("dP_train_u.npz")["dP_train_u"][:ntrain, :, :, -1].reshape(ntrain, 96*200).astype(np.float32)
    
#     x_test_branch = np.load("dP_train_a.npz")["dP_train_a"][-ntest:, :, :, :-2].transpose(0,3,1,2).astype(np.float32)
#     x_test = (x_test_branch, xrt)
    
#     y_test = np.load("dP_train_u.npz")["dP_train_u"][-ntest:, :, :, -1].reshape(ntest, 96*200).astype(np.float32)
    
#     return x_train, y_train, x_test, y_test   

def get_data(ntrain, ntest):
    z = np.linspace(0, 1, 96).astype(np.float32)
    #r = np.linspace(0, 1, 200).astype(np.float32)
    #grid_x = np.load("dP_train_a.npz")["dP_train_a"][0,0,:,-2]
    # xrt = np.array([[a, b] for a in z for b in r])
    
    x_train = np.load("dP_train_a.npz")["dP_train_a"][:ntrain, :, :, :4].transpose(0,3,1,2).astype(np.float32)
    x_train_MIO = np.load("dP_train_a_MIO.npy")[:ntrain,-5:].astype(np.float32)
    grid_x = np.load("dP_train_a.npz")["dP_train_a"][0,0,:,-2]
    xrt = np.array([[a, b] for a in z for b in grid_x])
    x_train = (x_train,x_train_MIO, xrt)
    
    y_train = np.load("dP_train_u.npz")["dP_train_u"][:ntrain, :, :, -1].reshape(ntrain, 96*200).astype(np.float32)
    
    #x_test = np.load("dP_test_a.npz")["dP_test_a"][-ntest:, :, :, :3].transpose(0,3,1,2).astype(np.float32)
    x_test = np.load("dP_train_a.npz")["dP_train_a"][-ntest:, :, :, :4].transpose(0,3,1,2).astype(np.float32)
    #x_test_MIO = np.load("dP_test_a_MIO.npy")[:ntest].astype(np.float32)
    x_test_MIO = np.load("dP_train_a_MIO.npy")[-ntest:,-5:].astype(np.float32)
    x_test = (x_test,x_test_MIO, xrt)
    
    #y_test = np.load("dP_test_u.npz")["dP_test_u"][-ntest:, :, :, -1].reshape(ntest, 96*200).astype(np.float32)
    y_test = np.load("dP_train_u.npz")["dP_train_u"][-ntest:, :, :, -1].reshape(ntest, 96*200).astype(np.float32)
    #y_test = y_test - 4.172939172019009
    #y_test = y_test / 18.772821433027488
    
    return x_train, y_train, x_test, y_test, grid_x

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

    
class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return func.relu(y + x)


def net():
    ret_net = nn.Sequential(#U_net(9,9,9,0),
                            nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=7, stride=2, padding=1))
    ret_net.add_module("resnet_block1", resnet_block(32, 32, 2))
    ret_net.add_module("resnet_block2", resnet_block(32, 64, 2))
    ret_net.add_module("resnet_block3", resnet_block(64, 128, 2))
    ret_net.add_module("resnet_block4", resnet_block(128, 256, 2))
    ret_net.add_module("resnet_block5", resnet_block(256, 512, 2))
    ret_net.add_module("global_ave_pool", GlobalAvgPool2d())
    ret_net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, pca.n_components_)))

    return ret_net

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        """
        The forward function.
        """
        return func.avg_pool2d(x, kernel_size=x.size()[2:])

x_train, y_train, x_test, y_test, grid_x = get_data(4000, 500)

grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
grid_dx = torch.as_tensor(grid_dx).reshape(1,1,198)

data = dde.data.QuadrupleCartesianProd(x_train, y_train, x_test, y_test)
#data = dde.data.QuadrupleCartesianProd(x_test, y_test, x_test, y_test)

pca = PCA(n_components=0.9999).fit(y_train.reshape(4000,96*200))



def output_transform(inputs, outputs):
    return outputs / torch.square(torch.as_tensor(pca.n_components_)) + torch.as_tensor(pca.mean_)
    

Net = dde.nn.pytorch.mionet.MIONetCartesianProd(
        [4000*96*200,net()],[3,300,300,300,pca.n_components_],[2,300,300,pca.n_components_],
        {"branch1":"relu",
        "branch2":"relu",
        "trunk":"relu"},
        "Glorot normal",
        ("l2",1e-4),
        True
    )


Net = dde.nn.pytorch.mionet.PODMIOPortNet(
        pod_basis=pca.components_.T * np.sqrt(200*96),
        layer_sizes_branch1=[4000*96*200,net()],layer_sizes_branch2=[5,300,300,pca.n_components_],layer_sizes_port=[2*pca.n_components_,300,300,pca.n_components_],
        activation={"branch1":"relu",
        "branch2":"relu",
        "trunk":"relu",
        "port":"relu"},
        connect_method="cat",
        kernel_initializer="Glorot normal",
	layer_sizes_trunk=None,
        regularization=None,
        trunk_last_activation=True)
Net = dde.nn.pytorch.mionet.PODMIONet(
        pca.components_.T * np.sqrt(200*96),
        [4000*96*200,net()],[5,300,300,pca.n_components_],
        {"branch1":"relu",
        "branch2":"relu",
        "trunk":"relu"},
        "Glorot normal",
        None,
        None,
        True
    )

Net.apply_output_transform(output_transform)

# def loss_fnc(y_true, y_pred):
#         size = y_true.shape[0]
#         mask = y_true.reshape(size,96,200)[:,:,0]!=-0.2222862
#         thickness = mask.sum(axis=1)
#         loss = torch.norm(y_true.reshape(size, 96, 200)[:, :thickness[0], :].reshape(size,thickness[0]*200)-y_pred.reshape(size, 96, 200)[:, :thickness[0], :].reshape(size,thickness[0]*200),dim=1)/ torch.norm(y_true.reshape(size, 96, 200)[:, :thickness[0], :].reshape(size,thickness[0]*200), dim=1)
#         for i in range(1,size):
#             loss = torch.cat((loss,torch.norm(y_true.reshape(size, 96, 200)[:, :thickness[i], :].reshape(size,thickness[i]*200)-y_pred.reshape(size, 96, 200)[:, :thickness[i], :].reshape(size,thickness[i]*200),dim=1)/ torch.norm(y_true.reshape(size, 96, 200)[:, :thickness[i], :].reshape(size,thickness[i]*200), dim=1)))
#         return torch.mean(loss)

def loss_fnc(y_true, y_pred):
    size = y_true.shape[0]
    y_true = y_true.reshape(size,96,200)
    y_pred = y_pred.reshape(size,96,200)
    x = torch.max(y_true,axis=2)[0].isclose(torch.tensor(0.0))
    y = torch.min(y_true,axis = 2)[0].isclose(torch.tensor(0.0))
    mask = y.logical_and(x)
    mask = 1-mask.to(torch.float32)
    mask = mask.reshape((size,96,1))
    y_true = y_true*mask
    y_pred = y_pred*mask
    dydx_true = (y_true[:,:,2:]-y_true[:,:,:-2])/grid_dx
    dydx_pred = (y_pred[:,:,2:]-y_pred[:,:,:-2])/grid_dx
    y_true = y_true.reshape(size,96*200)
    y_pred = y_pred.reshape(size,96*200)
    dydx_true = dydx_true.reshape(size,96*198)
    dydx_pred = dydx_pred.reshape(size,96*198)
    ori_loss =  torch.mean(torch.norm(y_true - y_pred, 2, dim=1) / torch.norm(y_true, 2, dim=1))
    #ori_loss =  torch.mean(torch.norm(y_true - y_pred, dim=1))
    der_loss =  torch.mean(torch.norm(dydx_true - dydx_pred, 2, dim=1) / torch.norm(dydx_true, 2, dim=1))
    return ori_loss #+ 0.5 * der_loss

def metrics_mae_mask(y_true, y_pred):
    size = y_true.shape[0]
    
    y_true = y_true.reshape(size,96,200)
    y_pred = y_pred.reshape(size,96,200)
    x = np.isclose(np.max(y_true,axis=2),-0.22228621)
    y = np.isclose(np.min(y_true,axis = 2),-0.22228621)
    mask = np.logical_and(x,y)
    mask = 1-mask.astype(np.float32)
    mask = mask.reshape((size,96,1))
    y_true = y_true*mask
    y_pred = y_pred*mask
    y_true = y_true.reshape(size,96*200)
    y_pred = y_pred.reshape(size,96*200)
    
    return np.mean(np.abs(y_true - y_pred)/np.abs(y_true))

def metrics_mae(y_true, y_pred):
    size = y_true.shape[0]
    
    return np.mean(np.abs(y_true - y_pred)/np.abs(y_true))
    
def metrics_Rsquare_mask(y_true,y_pred):

    
    size = y_true.shape[0]
    y_true = y_true.reshape(size,96,200)
    y_pred = y_pred.reshape(size,96,200)
    x = np.isclose(np.max(y_true,axis=2),-0.22228621)
    y = np.isclose(np.min(y_true,axis = 2),-0.22228621)
    mask = np.logical_and(x,y)
    mask = 1-mask.astype(np.float32)
    mask = mask.reshape((size,96,1))
    y_true = y_true*mask
    y_pred = y_pred*mask
    y_true = y_true.reshape(size,96*200)
    y_pred = y_pred.reshape(size,96*200)
    
    return 1-np.sum(np.square(y_true-y_pred))/np.sum(np.square(y_true-np.mean(y_true,axis = 1).reshape(size,1)))

def metrics_Rsquare(y_true,y_pred):

    
    size = y_true.shape[0]
    
    return 1-np.sum(np.square(y_true-y_pred))/np.sum(np.square(y_true-np.mean(y_true,axis = 1).reshape(size,1)))

def Rsquare(y_true,y_pred):
    size = y_true.shape[0]
    y_true = y_true.reshape(size,96,200)
    y_pred = y_pred.reshape(size,96,200)
    sse = 0
    sst = 0
    for i in range(size):
        z_axis = y_true[i,:,0]
        mask = np.isclose(z_axis,0.0)
        thickness = int(sum(1-mask.astype(np.float32)))
        y_true_i = y_true[i,:thickness,:]
        y_pred_i = y_pred[i,:thickness,:]
        sse = sse + np.sum(np.square(y_true_i.flatten()-y_pred_i.flatten()))
        sst = sst + np.sum(np.square(y_true_i.flatten()-np.mean(y_true_i.flatten())))
    return 1 - sse/sst

model = dde.Model(data, Net)
model.compile("adam", loss=loss_fnc,
                  lr=5e-4, decay=("step", 1000, 1), metrics=[Rsquare] )
                  
losshistory, train_state = model.train(epochs=100000, batch_size=200)

yhat = model.predict((x_test[0][0].reshape(1,9,96,200),x_test[1]))
model.save('pre_model')

mask = np.isclose(y_test.reshape(500,96,200)[0,:,0],-0.22228621)
thickness = int(sum(1-mask.astype(np.float32)))


plt.figure(figsize=(30,30))
y = y_test.reshape(500,96,200)[0,:thickness,:].flatten()
y_hat = yhat.reshape(96,200)[:thickness,:].flatten()
plt.plot(y,y_hat,".",np.linspace(0,y.max(),10000),np.linspace(-0.05,y.max()-0.05,10000),"-.",np.linspace(0,y.max(),10000),np.linspace(0.05,y.max()+0.05,10000),"-.")
#plt.plot(y_test[0],yhat.reshape(200*96),".")
plt.savefig("true_pred.png")
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

