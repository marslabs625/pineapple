from lib.metrics import Catgorical_Accuracy, Confusion_Matrix
from lib.tools import Timer
import torch

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    '''
    train a model in an epoch
    '''
    size = len(dataloader.dataset)
    current = 0
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss += loss

        train_acc += Catgorical_Accuracy(pred, y)

        current += len(X)
        if batch % 2 == 0:
            print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end = '\r')
    print()
    
    train_loss /= size / batch_size
    train_acc /= size
    print(f"Train Error: \n Accuracy: {(100*train_acc):.1f}%, Avg loss: {train_loss:8f} \n")

    return train_loss, train_acc


def test_loop(dataloader, model, loss_fn, batch_size, device, mode):
    '''
    evaluate model in validation stage or test stage
    '''
    size = len(dataloader.dataset)
    test_loss, test_acc = 0, 0
    confusion_matrix = Confusion_Matrix(4)
    model.eval()

    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            test_acc += Catgorical_Accuracy(pred, y)
            confusion_matrix.collect(pred, y)

    test_loss /= size / batch_size
    test_acc /= size
    print(f"{mode} Error: \n Accuracy: {(100*test_acc):.1f}%, Avg loss: {test_loss:8f} \n")

    confusion_matrix.result()

    return test_loss, test_acc

def fit(epochs, train_dataloader, model, loss_fn, optimizer, batch_size, device, 
        val_dataloader=None, early_stop=None):
    '''
    train a model with cpu or gpu device
    '''
    timer = Timer(epochs)
    optimizer_n = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1, verbose=True)
    stop_count = 0
    best_loss = None

    history = {}
    history['epoch'] = []
    history['train_loss'] = []
    history['train_acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []

    for epoch in range(epochs):
        timer.start()
        optimizer.step()
        print(f"Epoch {epoch+1}\n----------------------------------------")
        optimizer_n.step()
        history['epoch'].append(epoch + 1)

        train_history = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
        history['train_loss'].append(train_history[0])
        history['train_acc'].append(train_history[1])

        if val_dataloader:
            val_history = test_loop(val_dataloader, model, loss_fn, batch_size, device, 'Validation')
            history['val_loss'].append(val_history[0])
            history['val_acc'].append(val_history[1])

            if best_loss and best_loss < history['val_loss'][-1]:
                stop_count += 1

            else:
                best_weights = model.state_dict()
                best_loss = history['val_loss'][-1]
                stop_count = 0
            
            if stop_count == early_stop:
                model.load_state_dict(best_weights)
                break

        timer.finish()

        print(f"Time: {timer.elapsed_time:.2f} sec, ETA: {timer.ETA}\n")

    return history

def evaluate(test_dataloader, model, loss_fn, batch_size, device):
    '''
    evaluate a model with cpu or gpu device
    '''
    timer = Timer()
    timer.start()

    test_result = test_loop(test_dataloader, model, loss_fn, batch_size, device, 'Test')

    results = {}
    results['epoch'] = ['test']
    results['test_loss'] = [test_result[0]]
    results['test_acc'] = [test_result[1]]


    timer.finish()

    print(f"Time: {timer.elapsed_time:.2f} sec\n")

    return results

def replace_weights(model1, model2, first_layer, last_layer):
    '''
    load pretrained model weights to custom model
    '''
    model1_weights = model1.state_dict()
    model2_weights = model2.state_dict()
    model1_weights = [w for w in model1_weights.values()]

    i = 0
    j = 0
    for k in model2_weights.keys():
        if first_layer <= j < last_layer:
            model2_weights[k] = model1_weights[i]
            i += 1
        j += 1

    model2.load_state_dict(model2_weights)

    return model2