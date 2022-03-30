import torch
import torch.nn as nn

# pool of square window of size=3, stride=2
#m = nn.AvgPool2d(kernel_size=2, stride=2)
#t = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = 3)
bn1 = nn.BatchNorm2d(64)
maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
input = torch.randn(256, 3, 32, 32)
output = conv1(input)
print(output.shape)
output2 = bn1(output)
print(output2.shape)
output3 = maxpool1(output2)
print(output3.shape)