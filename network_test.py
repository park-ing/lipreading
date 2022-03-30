import torch
from torch import nn
from torch.utils.data import DataLoader # Dataset을 순회 가능한 객체로 감싼다.
from torchvision import datasets  # 샘플과 정답(label)을 저장
from torchvision.transforms import ToTensor
from TSM_resnet18 import ResNet, BasicBlock

''' Fasion MNIST 사용
모든 TorchVision Dataset은 샘플과 정답을 각각 변경하기 위한 
transform과 target_transform의 두 인자를 포함한다.
'''

# 공개 데이터셋에서 학습 데이터 다운
train_data = datasets.CIFAR10(
    root="data1",
    train=True,
    download=True,
    transform=ToTensor(), 
)

# 공개 데이터셋에서 테스트 데이터를 다운
test_date = datasets.CIFAR10(
    root="data1",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Dataset을 DataLoader의 인자로 전달한다. 
# 이는 데이터셋을 순회 가능한 객체(iterble)로 감싸고, 자동화된 배치, 샘플링, 섞기 및
# 다중 프로세스로 데이터 불러오기(multiprocess data leading)를 지원한다.
# Dataloader 객체의 각 요소는 (64)개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환한다.
batch_size = 256

# 데이터로더 생성
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_date, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
'''
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
'''

# PyTorch에서 신경망 모델은 nn.Module을 상속받는 클래스(class)를 생성하여 정의한다.
# __init__함수에서 신경망의 계층(layer)들을 정의하고 forward 함수에서 신경망에 데이터를 어떻게 전달할 지 지정
# 가능한 경우 GPU로 신경망을 이동시켜 연산을 가속

# 학습에 사용할 CPU나 GPU 장치를 얻는다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


'''
RESNET18
'''
model = ResNet(BasicBlock, [2,2,2,2])
#print(model)
model.to(device)

# 모델 매개변수 최적화하기
# 모델을 학습하려면 손실 함수(loss function)와 옵티마이저(optimizer)가 필요하다.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# 모델의 성능 확인
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # inference를 위해 layer을 바꾸려고 사용한다.
    test_loss, correct = 0, 0
    with torch.no_grad():   # 학습X, autograd 엔진 종료, 역전파에 필요한 메모리 등을 절약 가능.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")






'''
# 모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 모델 불러오기
# 모델 구조를 다시 만들고 상태 사전을 모델에 불러오는 과정이 포함된다.
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# 모델을 사용하여 예측
classes = [
        "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_date[0][0], test_date[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
'''