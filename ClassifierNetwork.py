import os
import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict


class ClassifierNetwork(object):

    #Intialize the classifier network.

    def __init__(self,a_gpu,lr,model_name,hidden_units):

        # Hyperparameters
        self.nn_entropy = nn.CrossEntropyLoss()
        # Building the model.
        self.model = self.create_model(model_name,hidden_units)
        # Stochastic gradient descent..
        self.nn_optimizer = optim.SGD(self.model.classifier.parameters(), lr=float(lr))

        # Device selection.
        if a_gpu :
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
           self.device = "cpu"

        #Move the model to respective device
        self.model.to(self.device)

    def create_model(self, model_name, hidden_units):
        """
        Builds the model based on
        model name and number of hidden layers parameters.
        """
        try:
           self.model = getattr(models, str(model_name))(pretrained=True)
           print(self.model)
        except:
           raise ValueError('Invalid model name ' + model_name)

        for param in self.model.parameters():
           param.requires_grad = False

        print("feature_Set")
        print(self.model.classifier[0].in_features)

        self.model.classifier = nn.Sequential(OrderedDict([
           ('fc1', nn.Linear(int(self.model.classifier[0].in_features), int(hidden_units))),
           ('relu', nn.ReLU()),
           ('dropout', nn.Dropout(p=0.5)),
           ('fc2', nn.Linear(int(hidden_units), 102)),
           ('output', nn.LogSoftmax(dim=1))
        ]))

        return self.model


    def train(self, input_t,labels_t, epoch, total, running_loss):
        """
         Training the model using training dataloader.
        """

        self.model.train()

        #print("Device selected.{}".format(self.device))
        # Move the data to either GPU or CPU
        inputs, labels = input_t.to(self.device), labels_t.to(self.device)

        # Intialize the gradient for the variables to zero.
        self.nn_optimizer.zero_grad()

	    # Forward Pass
        output = self.model.forward(inputs)

        # Calculate loss
        loss = self.nn_entropy(output, labels)

        # Backpropagation.
        loss.backward()
        # Accumulate the backpropagation gradients.
        self.nn_optimizer.step()

        # Training loss accumulation.
        running_loss += loss.item()

        return running_loss


    def validate(self, input_v, labels_v, epoch, total, running_loss,accuracy):

        """
        Validate the model after each training phase.
        """
        # Set evaulation mode of the model.
        self.model.eval()

        # move the variables to GPU/CPU
        inputs,labels = input_v.to(self.device), labels_v.to(self.device)

        # initialize the gradient to zero.
        self.nn_optimizer.zero_grad()

        # Feed forward the model.
        output = self.model.forward(inputs)
        loss = self.nn_entropy(output, labels)

        running_loss += loss.item()
        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return running_loss, accuracy


    def save_checkpoint(self,data_loader,path,lr,epochs,arch):
        """
        Save a model checkpoint in a path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint_path = os.path.join(path,'checkpoint.pth')
        self.model.class_to_idx = data_loader.dataset.class_to_idx

        checkpoint = {
            'model': self.model,
            'output_size': 102,
            'input_size': 25088,
            'state_dict': self.model.state_dict(),
            'classifier': self.model.classifier,
            'epochs':epochs,
            'arch':arch,
            'batch_size':64,
            'learning_rate':lr,
            'class_to_idx': self.model.class_to_idx,
            'optimizer_dict': self.nn_optimizer.state_dict(),
            'batch_size': data_loader.batch_size,
        }
        print("Saving checkpoint")
        torch.save(checkpoint, checkpoint_path)
