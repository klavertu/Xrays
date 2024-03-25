import copy
import os, glob
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage.io import imread
from torchsummary import summary
from numpy import genfromtxt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report




argslr = 0.00001
argsDataAug = True
argsBatchSize = 128
argsTestBatchSize = 1000
argsNepochs = 60
argsSave = './experiment1'
argsGpu = 0

args = (argslr,argsDataAug,argsBatchSize,argsTestBatchSize,argsNepochs,argsSave,argsGpu)


def norm(dim):
    return nn.BatchNorm1d(dim)
    #return nn.BatchNorm2d(dim)


device = torch.device('cuda:' + str(argsGpu) if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
    

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val    
    

    
def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
   
    

def get_loaders(data_aug=False, batch_size=16, test_batch_size=1000, perc=1.0):
    
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    train_set = datasets.ImageFolder('./chest_xray/train', transform=transform)
    val_set = datasets.ImageFolder('./chest_xray/test', transform=transform)
    
    print('Train data size: ', len(train_set))
    print('Test data size: ', len(val_set))
    
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    train_loader = data.DataLoader(dataset=train_set,
                                      batch_size=batch_size,
                                      shuffle=True)
    
    test_loader = data.DataLoader(dataset=val_set,
                                      #batch_size=batch_size,
                                      batch_size=256,
                                      shuffle=True)



    return train_loader, test_loader



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = argslr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn



def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 2)
        pred = model(x).cpu().detach().numpy()
        
        score_f1 = f1_score(y, pred)
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(pred, axis=1)
        total_correct += np.sum(predicted_class == target_class)
        

    return (total_correct / len(dataset_loader.dataset)), score_f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    
    
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger 
    
    
    
def precision_recall_f1(labels, preds):


    return precision_score(np.array(labels), np.array(preds), average="weighted", labels=np.unique(np.array(preds))),recall_score(np.array(labels), np.array(preds), average="weighted", labels=np.unique(np.array(preds))),f1_score(np.array(labels), np.array(preds), average="weighted", labels=np.unique(np.array(preds)))

    
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1 = 0.0
            running_precision = 0.0
            
            running_preds = []
            running_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs1, aux_outputs2 = model(inputs)
                        #print(outputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs1, labels)
                        loss3 = criterion(aux_outputs2, labels)
                        loss = loss1 + 0.3*loss2 + 0.3*loss3
                    else:
                        outputs = model(inputs)
                        #print(outputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_f1 += precision_recall_f1(labels.cpu().data, preds.cpu())
                running_preds.extend(preds.cpu().numpy().tolist())
                running_labels.extend(labels.cpu().data.numpy().tolist())
   

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            #epoch_f1 = running_f1 / len(dataloaders[phase].dataset)
            epoch_f1 = precision_recall_f1(running_labels, running_preds)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("Precision, Recall, F1: ",epoch_f1)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                history['train_acc'].append(epoch_acc)
                history['train_loss'].append(epoch_loss)
            elif phase == 'val':
                history['val_acc'].append(epoch_acc)
                history['val_loss'].append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
    
if __name__ == '__main__':
    
    makedirs(argsSave)
    logger = get_logger(logpath=os.path.join(argsSave, 'logs'), filepath=os.path.abspath(argsSave+'/logs'))
    logger.info(args)

    print(device)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True, aux_logits=True)
    
    print(model)
    
    model.aux1.fc2 = nn.Linear(1024, 2)
    model.aux2.fc2 = nn.Linear(1024, 2)
    model.fc = nn.Linear(1024, 2)

    print(model)
    
    print('Number of parameters: {}'.format(count_parameters(model)))
    
    #model.fc = torch.nn.Linear(num_ftrs, 2)
    model.eval()
    
    
    model = model.to(device)
    params_to_update = model.parameters()
    print("Params to learn:")

    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
            
    #optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = torch.optim.Adam(params_to_update, lr=argslr)

    criterion = torch.nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    
    train_loader, val_loader = get_loaders(batch_size=argsBatchSize)

    
    model, hist = train_model(
        model,
        dataloaders={
            "train": train_loader,
            "val": val_loader
        },
        criterion=criterion,
        optimizer=optimizer_ft,
        num_epochs=argsNepochs,
        is_inception=True
    )
  
    
