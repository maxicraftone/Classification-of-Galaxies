import torch
from tqdm import tqdm

def train_validation_loop(trainloader, testloader, optimizer, criterion, net, epochs=1):
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

        #Validation
        net.eval()

        test_loss = 0
        accuracy = 0

        testloop = tqdm(testloader)

        for i, (images,labels_t01,labels_t02) in enumerate(testloop):
            with torch.no_grad():
                test_predictions = net(images)
                    
                test_loss += criterion(test_predictions, labels_t01)
                                
                top_probability, top_guess = test_predictions.topk(1, dim=1) #Tensor of top guesses for each image
                guess_correct = top_guess == labels_t01.view(top_guess.shape) #Tensor of booleans for each guess (True: correct guess, False: wrong guess)
                accuracy += torch.mean(guess_correct.type(torch.FloatTensor)) #Calculate mean of booleans (True:=1.0, False:=0.0)
                    
            testloop.set_description(f"> Validation on test data ")
            testloop.set_postfix(loss=test_loss.item()/(i+1),acc=accuracy.item()/(i+1))
    
    return training_losses


def trainloop(trainloader, optimizer, criterion, net, epochs=1):
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

def save_net(net):
    torch.save(net.state_dict(), net.save_file)

def load_net(net, device='cpu'):
    net.load_state_dict(torch.load(net.save_file, map_location=torch.device(device)))
    return net

def validation_loop(testloader, criterion, net):
    #Validation
    net.eval()

    test_loss = 0
    accuracy = 0

    testloop = tqdm(testloader)

    for i, (images,labels_t01,labels_t02) in enumerate(testloop):
        with torch.no_grad():
            test_predictions = net(images)
                
            test_loss += criterion(test_predictions, labels_t01)
                            
            top_probability, top_guess = test_predictions.topk(1, dim=1) #Tensor of top guesses for each image
            guess_correct = top_guess == labels_t01.view(top_guess.shape) #Tensor of booleans for each guess (True: correct guess, False: wrong guess)
            accuracy += torch.mean(guess_correct.type(torch.FloatTensor)) #Calculate mean of booleans (True:=1.0, False:=0.0)
                
        testloop.set_description(f"> Validation on test data ")
        testloop.set_postfix(loss=test_loss.item()/(i+1),acc=accuracy.item()/(i+1))