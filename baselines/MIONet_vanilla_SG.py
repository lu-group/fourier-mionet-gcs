import time
import numpy as np
from baseline_networks import *


def get_data(ntrain, ntest):
    z = np.linspace(0, 1, 96).astype(np.float32)
    t = np.linspace(0, 1, 24).astype(np.float32)
    r = np.linspace(0, 1, 200).astype(np.float32)
    xrt = np.array([[a, b, c] for a in t for b in z for c in r])
    xrt = xrt.astype(np.float32)

    field_input = [True, True, True, False, False, False, False, False, False, True, True]
    x_train = np.load("sg_train_a.npz")["sg_train_a"][:ntrain, :, :, field_input].transpose(0,1,2,3).astype(np.float32)
    x_train_MIO = np.load("sg_train_a_MIO.npy")[:ntrain, :].astype(np.float32)
    grid_x = np.load("sg_train_a.npz")["sg_train_a"][0, 0, :, -2].astype(np.float32)
    x_train = (x_train, x_train_MIO, xrt)
    y_train = np.load("sg_train_u.npz")["sg_train_u"][:ntrain, :, :, :].transpose(0, 3, 1, 2).reshape(ntrain,24 * 96 * 200).astype(np.float32)
    x_test = np.load("sg_test_a.npz")["sg_test_a"][-ntest:, :, :, field_input].transpose(0, 1, 2, 3).astype(np.float32)
    x_test_MIO = np.load("sg_test_a_MIO.npy")[-ntest:, :].astype(np.float32)
    x_test = (x_test, x_test_MIO, xrt)
    y_test = np.load("sg_test_u.npz")["sg_test_u"][-ntest:, :, :, :].transpose(0, 3, 1, 2).reshape(ntest,24 * 96 * 200).astype(np.float32)

    return x_train, y_train, x_test, y_test, grid_x


x_train, y_train, x_test, y_test, grid_x = get_data(4500, 500)
data = dde.data.QuadrupleCartesianProd(x_train, y_train, x_test, y_test)

Net = MIONetCartesianProd(layer_sizes_branch1=[4500*96*200*3, Encoder()],
                          layer_sizes_branch2=[7, 512, 512, 512, 512],
                          layer_sizes_trunk=[3, 512, 512, 512, 512],
                          activation='relu',
                          kernel_initializer="Glorot normal",
                          regularization=("l2",4e-6),
                          trunk_last_activation=False,
                          merge_operation="mul",
                          layer_sizes_merger=None,
                          output_merge_operation="mul",
                          layer_sizes_output_merger=None)

z = np.linspace(0, 1, 96).astype(np.float32)
grid_dy = z[:-1]-z[1:]
grid_dy = grid_dy.reshape(95,1)
grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
grid_dx = torch.as_tensor(grid_dx).reshape(1,1,1,198)
pre_mask_train = np.isclose(y_train.reshape(y_train.shape[0],24,96,200)[:,-1,:,0],0.0)
pre_mask_test = np.isclose(y_test.reshape(y_test.shape[0],24,96,200)[:,-1,:,0],0.0)
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


def MAE(y_true,y_pred):
    size = y_true.shape[0]
    y_true = y_true.reshape(size,24,96,200)
    y_pred = y_pred.reshape(size,24,96,200)
    mae = 0
    for i in range(size):
        z_axis = y_true[i,-1,:,0]
        mask = np.isclose(z_axis,0.0)
        mask=1-mask
        mask=mask.astype(bool)
        y_true_i = y_true[i][:,mask,:]
        y_pred_i = y_pred[i][:,mask,:]
        mae += np.mean(np.abs(y_true_i.flatten()-y_pred_i.flatten()))
    return mae/size


path = f'./train_MIONet_SG/'
if not os.path.exists(path):
    os.mkdir(path)

model = dde.Model(data, Net)
model.compile("adam", loss=loss_fnc,loss_weights=[1,0.5],
                  lr=2e-4, decay=("step", 2250*2, 0.9), metrics=[Rsquare_plume_tegother,MAE] )
checker = dde.callbacks.ModelCheckpoint("train_MIONet_SG/MIONet_SG",save_better_only=False,period=2250,monitor="test loss")
start = time.time()
losshistory, train_state = model.train(epochs=112500, batch_size=6,timestep_batch_size=8,training_time_size=24,display_every=750,init_test=True,callbacks=[checker])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
end = time.time()
print("running time:",print(end-start))
print("num of parameters:",model.net.num_trainable_parameters())