import torch
from torch.autograd import Variable

def generate_cam(img, extractor, classifier, device):
    extractor.to(device)
    classifier.to(device)

    extractor.eval()
    classifier.eval()

    extractor.zero_grad()
    classifier.zero_grad()

    img = img.to(device)
    features = extractor(img)
    features = Variable(features, requires_grad=True)

    classes = classifier(features)
    pred_prob = classes[:, classes.argmax(1)]
    pred_prob.backward()

    grads = features.grad
    weights = grads.mean(axis=(2, 3))

    features = features.detach()
    cam = torch.zeros(features[0, 0, :, :].size()).to(device)
    for w, f in zip(weights[0], features[0]):
        cam += w * f

    cam = torch.maximum(cam, torch.zeros(1).to(device))

    return cam.to('cpu'), classes.argmax(1).to('cpu')