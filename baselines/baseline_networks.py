import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), relu_slop=0.2):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size,
                            stride=stride, padding=padding),
                  nn.LeakyReLU(relu_slop, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(Encoder, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), padding=(1, 1))
        self.convblock5_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock5_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(6, 7), padding=0)
        self.flatten1 = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(dim5, dim5*4)
        self.leakyrelu1 = torch.nn.LeakyReLU(0.2)
        self.linear2 = torch.nn.Linear(dim5*4, dim5)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, 96, 200, 5)
        x = x.permute(0, 3, 1, 2)
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock8(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.leakyrelu1(x)
        x = self.linear2(x)
        return x


# Modified from Deepxde
class MIONetCartesianProd(dde.nn.pytorch.NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=None,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = dde.nn.activations.get(activation["branch1"])
            self.activation_branch2 = dde.nn.activations.get(activation["branch2"])
            self.activation_trunk = dde.nn.activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = self.activation_merger = self.activation_output_merger = dde.nn.activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = dde.nn.pytorch.FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = dde.nn.pytorch.FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            if isinstance(activation, dict):
                self.activation_merger = dde.nn.activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = dde.nn.pytorch.FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            if isinstance(activation, dict):
                self.activation_output_merger = dde.nn.activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                # User-defined network
                self.output_merger = layer_sizes_output_merger[1]
            else:
                # Fully connected network
                self.output_merger = dde.nn.pytorch.FNN(
                    layer_sizes_output_merger,
                    self.activation_output_merger,
                    kernel_initializer,
                )
        else:
            self.output_merger = None
        self.trunk = dde.nn.pytorch.FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            y_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        # Dot product
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of merger net and trunk net do not match."
            )
        # output merger net
        if self.output_merger is None:
            y = torch.einsum("ip,jp->ij", y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == "mul":
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == "add":
                y = y_func + y_loc
            elif self.output_merge_operation == "cat":
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)
        # Add bias
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
