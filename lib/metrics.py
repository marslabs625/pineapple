import torch

def Binary_Accuracy(pred, label, threshold=0.5):
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    train_acc = (pred == label).sum().item()

    return train_acc

def Catgorical_Accuracy(pred, label):
    pred = pred.argmax(1)
    train_acc = (pred == label).sum().item()

    return train_acc

class Confusion_Matrix():
    def __init__(self, num_classes):
        self.c_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    def collect(self, pred, label):
        pred = pred.argmax(1)
        for p, l in zip(pred, label):
            self.c_matrix[l][p] += 1

    def result(self):
        print(f"{self.c_matrix}\n")