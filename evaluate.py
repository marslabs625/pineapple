from lib.dataset import Pineapple
from lib.models import VGG16
from lib.coreFunc import evaluate
from lib.cam import generate_cam
from torch.utils.data import DataLoader
from matplotlib import cm
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
from PIL import Image

model_name = 'vgg-test_auto_learning_rate_2'
weights_path = os.path.join('./weights', model_name)
results_path = os.path.join('./results', model_name)

test_batch_size = 1

#load data
data_dir = './data/wav'

test_label_path = os.path.join(data_dir, 'test.csv')
test_data = Pineapple(data_dir, "test")
test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

#assign model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print('========================================')

model = VGG16()

load_path = os.path.join(weights_path, os.path.split(weights_path)[1] + '.pth')
model_weights = torch.load(load_path)
model.load_state_dict(model_weights)

model = model.to(device)
#print(model)
print('========================================')

loss_fn = torch.nn.CrossEntropyLoss()

#testing
results = evaluate(test_dataloader, model, loss_fn, test_batch_size, device)
print('========================================')

#generate cam
extractor = torch.nn.Sequential(model.block1, model.block2, model.block3, model.block4, model.block5)
classifier = torch.nn.Sequential(model.flatten, model.classifier)
del model
torch.cuda.empty_cache()

f_id = 0
for batch, (X, y) in enumerate(test_dataloader):
    cam, pred = generate_cam(X, extractor, classifier, device)
    heatmap = np.resize(cam, np.shape(X[0, 0, :, :]))
    heatmap = heatmap - heatmap.min()
    heatmap = (heatmap * (255 / heatmap.max())).astype(np.uint8)
    jet = cm.get_cmap('jet')
    jet = jet(np.arange(256))[:, :3]
    heatmap = jet[heatmap]
    
    X = X[0, 0, :, :]
    # X = X - X.min()
    # plt.pcolor(X, cmap = cm.get_cmap('magma'))
    # plt.clim(X.min(), X.max())
    # plt.colorbar()
    # plt.show()
    # X = (X * (255 / X.max())).numpy().astype(np.uint8)
    # jet = cm.get_cmap('magma')
    # jet = jet(np.arange(256))[:, :3]
    # X = jet[X]

    # output_img = heatmap * 0.8 + X
    # output_img = output_img * (255 / output_img.max())
    # output_img = output_img.astype(np.uint8)

    if not os.path.isdir(os.path.join(results_path, 'cam')):
        os.mkdir(os.path.join(results_path, 'cam'))

    fig, axs = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)

    axs[0].set_title('Mel Spectrogram', fontsize=20)
    axs[0].set_ylabel('Frequency', fontsize=18)
    axs[0].set_xlabel('Time', fontsize=18)
    mesh = axs[0].pcolormesh(X, cmap = cm.get_cmap('magma'))
    mesh.set_clim(X.min(), X.max())
    # axs[0].imshow(X, aspect='auto')
    axs[0].set_ylim(0, np.shape(X)[0])
    axs[1].set_title('Grad-CAM (Heatmap)', fontsize=20)
    axs[1].set_xlabel('Time', fontsize=18)
    axs[1].imshow(heatmap, aspect='auto')
    axs[1].set_ylim(0, np.shape(heatmap)[0])

    plt.colorbar(mesh, ax=axs[0]).ax.set_title('dB')
    plt.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('jet')), ax=axs[1])
    # plt.ylim(0, np.shape(output_img)[0])

    filename = test_data.labels.index[f_id]
    filename = filename.split('.')[0]
    plt.savefig(f"{os.path.join(results_path, 'cam')}/{filename}-{pred[0].numpy()}{y[0].numpy()}.png", format='png')
    # plt.show()
    plt.close()
    f_id += 1