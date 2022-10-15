from lib.dataset import Pineapple
from lib.models import VGG16
from lib.coreFunc import fit, evaluate, replace_weights
from lib.tools import Plotter
from lib.random import Random
#from lib.random_csv import Random
from torch.utils.data import DataLoader
import torch
from torchvision.models import vgg16
import os
import pandas as pd

model_name = 'weight_decay=1e-3_learning_rate=1e-5_4'
weights_path = os.path.join('./weights', model_name)
results_path = os.path.join('./results', model_name)

train_batch_size = 16
test_batch_size = 16
learning_rate = 1e-5
weight_decay = 1e-3
epochs = 1000
early_stop = 50

plotter_x_interval = 100

#load data
data_dir = './data/wav'
data = Pineapple(data_dir, "train")
#print(len(data))
training_data, val_data, test_data = Random(data)
#training_data, val_data, test_data = Random(data, data_dir) #csv
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=train_batch_size)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

# if __name__ == '__main__': ##if workers != 0
#assign model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using {} device'.format(device))
print('========================================')

pretrained_model = vgg16(pretrained=True)
model = VGG16()

model = replace_weights(pretrained_model, model, 2, 28)

model = model.to(device)
#print(model)
print('========================================')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#1e-4(64.8&66.5)
#1e-5(56)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()

#training
history = fit(epochs, train_dataloader, model, loss_fn, optimizer, train_batch_size, device, val_dataloader, early_stop)

print('==============train + val===============')

#train + validation
train_dataloader = DataLoader(training_data + val_data, batch_size=train_batch_size, shuffle=True)
history_tv = fit(100, train_dataloader, model, loss_fn, optimizer, train_batch_size, device)

print('========================================')

results = evaluate(test_dataloader, model, loss_fn, test_batch_size, device)

if not os.path.isdir(os.path.dirname(weights_path)):
    os.mkdir(os.path.dirname(weights_path))

if not os.path.isdir(weights_path):
    os.mkdir(weights_path)

if not os.path.isdir(os.path.dirname(results_path)):
    os.mkdir(os.path.dirname(results_path))

if not os.path.isdir(results_path):
    os.mkdir(results_path)

save_path = os.path.join(weights_path, os.path.split(weights_path)[1] + '.pth')
torch.save(model.state_dict(), save_path)

del(history_tv['val_loss'])
del(history_tv['val_acc'])
log_df = pd.DataFrame.from_dict(history)
log_tv_df = pd.DataFrame.from_dict(history_tv)
results_df = pd.DataFrame.from_dict(results)
log_df = pd.concat([log_df, log_tv_df, results_df])
log_df.set_index('epoch', inplace=True)

log_df.to_csv(os.path.join(results_path, 'log.csv'))

#plot training history
plotter = Plotter()

savefig_path = os.path.join(results_path, 'loss.png')

plotter.append_data(history['train_loss'], 'b-', 'training')
if history['val_loss']:
    plotter.append_data(history['val_loss'], 'r-', 'validation')
plotter.line_chart(savefig_path, 'Loss', 'Epochs', 0, history['epoch'][-1], plotter_x_interval)

plotter.reset_dataList()

savefig_path = os.path.join(results_path, 'accuracy.png')

plotter.append_data(history['train_acc'], 'b-', 'training')
if history['val_acc']:
    plotter.append_data(history['val_acc'], 'r-', 'validation')
plotter.line_chart(savefig_path, 'Accuracy', 'Epochs', 0, history['epoch'][-1], plotter_x_interval)
