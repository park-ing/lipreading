import torch
import torch.nn as nn

# pool of square window of size=3, stride=2
#m = nn.AvgPool2d(kernel_size=2, stride=2)
#t = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = 3)
bn1 = nn.BatchNorm2d(64)
maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
input = torch.randn(3, 3, 2, 2)

#x = input
#nt, c, h, w = x.size()
#n_batch = nt // 3
##x = x.view(n_batch, 3, c, h, w)
#output = conv1(input)
#print(output.shape)
#output2 = bn1(output)
#print(output2.shape)
#output3 = maxpool1(output2)
#print(output3.shape)
'''
x = torch.randn(4, 3, 5, 5)
print(x)
nt, c, h, w = x.size()
n_batch = nt // 2
x = x.view(-1, 2, c, h, w)
out = torch.zeros_like(x)
fold = 1
out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
print('shift: {}'.format(out.shape))
print(out)
'''


def shift(x, n_segment, fold_div=3, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    print("---shift---")
    print("shift shape",x.shape)
    fold = c // fold_div
    if inplace:
        # Due to some out of order error when performing parallel computing. 
        # May need to write a CUDA kernel.
        raise NotImplementedError  
        # out = InplaceShift.apply(x, fold)
    else:
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        print("shift start")
        print(out)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        print(out)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        print(out)
        print("---merge---")

    return out.view(nt, c, h, w)

print(input.shape)
print(input)
x = shift(input,n_segment=3,fold_div=3,inplace=False)
print(x)
print(x.shape)


#-------------------------
inputt = torch.randn(32, 1, 28, 88, 88)
print(inputt.shape)
front_3d = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                        nn.BatchNorm3d(64),
                        nn.PReLU(num_parameters=64),
                        nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

tt = front_3d(inputt)
print(tt.shape)     
n_batch, n_channels, s_time, sx, sy = tt.shape
tt = tt.transpose(1, 2)
print(tt.shape)
tt = tt.reshape(n_batch*s_time, n_channels, sx, sy)     
print(tt.shape)
