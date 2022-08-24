from lib.dataset import Pineapple
from lib.models import VGG16
from lib.coreFunc import fit, evaluate, replace_weights
from lib.tools import Plotter
from torch.utils.data import DataLoader, random_split
import torch
from torchvision.models import vgg16
import os
import pandas as pd

model_name = 'vgg-test'
weights_path = os.path.join('./weights', model_name)
results_path = os.path.join('./results', model_name)

train_batch_size = 8
test_batch_size = 8
learning_rate = 1e-7
epochs = 10
early_stop = 50

plotter_x_interval = 100

#load data
data_dir = './data/wav/dictionary'
train_label_path = os.path.join(data_dir, 'train.csv')
val_label_path = os.path.join(data_dir, 'validation.csv')
test_label_path = os.path.join(data_dir, 'test.csv')

data = Pineapple(train_label_path, data_dir)
print(len(data))
training_data, val_data, test_data = random_split(data, [56, 8, 8], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=train_batch_size)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

# if __name__ == '__main__': ##if workers != 0
#assign model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('Using {} device'.format(device))
print('========================================')

pretrained_model = vgg16(pretrained=True)
model = VGG16()

model = replace_weights(pretrained_model, model, 2, 28)

model = model.to(device)
print(model)
print('========================================')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()

#training
history = fit(epochs, train_dataloader, model, loss_fn, optimizer, train_batch_size, device, val_dataloader, early_stop)
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

log_df = pd.DataFrame.from_dict(history)
results_df = pd.DataFrame.from_dict(results)
log_df = pd.concat([log_df, results_df])
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
