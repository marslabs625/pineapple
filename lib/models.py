from torch import nn
from torchvision.models import resnet50

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(), 
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), 
            nn.ReLU(), 
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(), 
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(), 
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128*67*64, 512), 
            nn.Linear(512, 512), 
            nn.Linear(512, 4), 
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        outputs = self.classifier(x)

        return outputs

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(28672, 4096),
            #nn.Linear(8*4*512, 69632),
            nn.Linear(4096, 4096), 
            nn.Linear(4096, 4), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        outputs = self.classifier(x)

        return outputs

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(8*4*512, 4096), 
            nn.Linear(4096, 4096), 
            nn.Linear(4096, 4), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        outputs = self.classifier(x)

        return outputs

class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.resnet50 = resnet50(pretrained=pretrained)
        self.resnet50.fc = nn.Linear(2048, 4)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet50(x)
        outputs = self.softmax(x)

        return outputs