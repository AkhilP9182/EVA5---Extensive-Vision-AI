import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.notebook as tq

def train(model, device, train_loader, optimizer, epoch, L1_param=0,L2_param=0, cyclicLR=False):
    model.train()
    pbar = tq.tqdm(train_loader,leave=False)
    correct,processed = 0,0

    if L2_param > 0:
        optimizer.param_groups[0]['weight_decay'] = L2_param
    
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # get samples
        optimizer.zero_grad()
        y_pred = model(data)   # The feed forward pass for generating output from data
        loss = criterion(y_pred, target)

        if (L1_param > 0):
            sum_weights = 0
            for param in model.parameters():
                sum_weights += torch.sum(abs(param))
            loss += (L1_param * sum_weights)
        
        loss.backward()         # Calculates the d(loss)/dx for each parameter x, which are accumulated into x.grad
        optimizer.step()        # optimizer.step() multiples the learing rate with the x.grad and updates each model parameter

        if cyclicLR:
            scheduler.step() # NOrmally scheduler.step() is called in the main code, but this scheduler requires ITERATIONS as steps
        
        # Update pbar-tqdm
        pred        = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct    += pred.eq(target.view_as(pred)).sum().item()
        processed  += len(data)
        acc         = 100*(correct/processed)
        string_name = 'Epoch={} | Batch={} | Loss={:.4f} | Acc={:.2f}'.format(epoch,batch_idx,loss.item(),acc)
        pbar.set_description(desc=string_name) # Updates the description at every timestep while showing the progress bar
    
    # ---- Obtaining Training Accuracy for EPOCH -------
    model.eval() 
    total_train_loss = 0
    correct_train    = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in train_loader:
            data, target   = data.to(device), target.to(device)
            output         = model(data)
            total_train_loss += criterion(output, target).item()     # sums up the loss for all samples in a batch               
            pred           = output.argmax(dim=1, keepdim=True)                      # get the index of the max log-probability
            correct_train += pred.eq(target.view_as(pred)).sum().item()              # Compare the predictions with the target
    
    train_loss     = total_train_loss/len(train_loader.dataset)
    train_accuracy = 100. * (correct_train / len(train_loader.dataset))
    print('\nEpoch:{} Learning Rate:{}\nTrain Set: Mean loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(epoch,
        optimizer.param_groups[0]['lr'], train_loss, correct_train, len(train_loader.dataset),train_accuracy))
    return train_loss,train_accuracy


def test(model, device, test_loader):
    model.eval()
    total_test_loss = 0
    correct_test    = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # torch.no_grad() disables gradient calculation during inference, thus reducing memory consumption
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output       = model(data)
            total_test_loss += criterion(output, target).item()     # sums up the loss for all samples in a batch  
            pred         = output.argmax(dim=1, keepdim=True)                      # get the index of the max log-probability
            correct_test+= pred.eq(target.view_as(pred)).sum().item()              # Compare the predictions with the target

    test_loss = total_test_loss/len(test_loader.dataset)
    test_accuracy = 100. * (correct_test / len(test_loader.dataset))
    print('Test Set : Mean loss: {:.4f}, Test Accuracy : {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_test, len(test_loader.dataset),test_accuracy))
    return test_loss,test_accuracy
