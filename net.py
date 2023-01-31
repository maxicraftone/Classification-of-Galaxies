import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import typing

def train_validation_loop(trainloader: DataLoader, valloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, net, epochs: int=1, backups: bool=False):
    """
    Trains the model on training data while also providing current accuracy on the validation data

    Parameters:
        trainloader (DataLoader): DataLoader containing the training dataset
        valloader (DataLoader): DataLoader containing the testing dataset
        optimizer (Optimizer): Optimizer to be used for updating model weights
        criterion (Loss): Loss function to calculate the loss to be minimized by the optimizer
        net (torch.nn.Module subclass): Model to be trained
        epochs (int): Number of training epochs
        backups (bool): Whether backups of the model state should be made every epoch
    """

    for epoch in range(epochs):
        #Training
        training_loss = 0

        trainloop = tqdm(trainloader)

        for i, data in enumerate(trainloop):
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data
            optimizer.zero_grad() #Set gradients to zero
            
            predictions = net(images)

            loss = criterion(predictions, labels)
            loss.backward()
            
            optimizer.step()
            
            training_loss += loss.item()

            trainloop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            trainloop.set_postfix(loss=training_loss/(i+1))

        if backups:
            save_net(net, filename=(net.save_file + '.epoch' + str(epoch)))

        #Validation
        net.eval()

        test_loss = 0
        accuracy = 0

        testloop = tqdm(valloader)

        for i, data in enumerate(testloop):
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data
            with torch.no_grad():
                test_predictions = net(images)
                    
                test_loss += criterion(test_predictions, labels)
                                
                _, top_guess = test_predictions.topk(1, dim=1) #Tensor of top guesses for each image
                guess_correct = top_guess == labels.view(top_guess.shape) #Tensor of booleans for each guess (True: correct guess, False: wrong guess)
                accuracy += torch.mean(guess_correct.type(torch.FloatTensor)) #Calculate mean of booleans (True:=1.0, False:=0.0)
                    
            testloop.set_description(f"> Validation on test data ")
            testloop.set_postfix(loss=test_loss.item()/(i+1),acc=accuracy.item()/(i+1))


def trainloop(trainloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, net: torch.nn.Module, epochs: int=1) -> typing.List[int]:
    training_losses = []

    for epoch in range(epochs):
        #Training
        training_loss = 0

        trainloop = tqdm(trainloader)

        for i, (images, labels_t01, labels_t02) in enumerate(trainloop):
            optimizer.zero_grad() #Set gradients to zero
            
            predictions = net(images)

            loss = criterion(predictions, labels_t01)
            loss.backward()
            
            optimizer.step()
            
            training_loss += loss.item()

            trainloop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            trainloop.set_postfix(loss=training_loss/(i+1))
        
        training_losses.append(training_loss/len(trainloader))
    
    return training_losses

def save_net(net, filename: str="") -> None:
    if filename != "":
        torch.save(net.state_dict(), filename)
    else:
        torch.save(net.state_dict(), net.save_file)

def load_net(net, filename:str="", device: str='cpu'):
    if filename != "":
        net.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    else:
        net.load_state_dict(torch.load(net.save_file, map_location=torch.device(device)))
    return net

def validation_loop(testloader: DataLoader, criterion: torch.nn.modules.loss._Loss, net: torch.nn.Module) -> None:
    #Validation
    net.eval()

    test_loss = 0
    accuracy = 0

    testloop = tqdm(testloader)

    for i, data in enumerate(testloop):
        if len(data) == 3:
            images, labels, _ = data
        else:
            images, labels = data
        with torch.no_grad():
            test_predictions = net(images)
                
            test_loss += criterion(test_predictions, labels)
                            
            _, top_guess = test_predictions.topk(1, dim=1) #Tensor of top guesses for each image
            guess_correct = top_guess == labels.view(top_guess.shape) #Tensor of booleans for each guess (True: correct guess, False: wrong guess)
            accuracy += torch.mean(guess_correct.type(torch.FloatTensor)) #Calculate mean of booleans (True:=1.0, False:=0.0)
                
        testloop.set_description(f"> Validation on test data ")
        testloop.set_postfix(loss=test_loss.item()/(i+1),acc=accuracy.item()/(i+1))