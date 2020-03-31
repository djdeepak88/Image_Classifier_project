import os
import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict


class ClassifierNetwork(object):

    #Intialize the classifier network.

    def __init__(self,a_gpu,lr,model_name,hidden_units):

        #Model input features
        if model_name == "vgg19":
            self.model_input_features = 25088
        elif model_name == "vgg16":
            self.model_input_features = 25088
        elif model_name == "alexnet":
            self.model_input_features = 9216
        elif model_name == "densenet121":
            self.model_input_features = 1024
        elif model_name == "resnet18":
            self.model_input_features = 512
        else:
            raise SystemExit('Not a valid model. The supported models are \n1.vgg19, \n2.vgg16, \n3.alexnet,\n4.densenet121,\n5.resnet18')

        self.model_name = model_name

        # Hyperparameters
        self.nn_entropy = nn.CrossEntropyLoss()

        # Initialize the custom classifier network.
        self.classifier_network = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(int(self.model_input_features), int(hidden_units))),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(int(hidden_units), int(hidden_units))),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(int(hidden_units), 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        # Building the model.
        self.model = self.create_model(hidden_units)

        # Stochastic gradient descent..
        if self.model_name in { "vgg16",  "vgg19",  "densenet121", "alexnet"}:
               self.nn_optimizer = optim.SGD(self.model.classifier.parameters(), lr=float(lr))

        # for alexnet and resnet18
        if self.model_name == "resnet18":
               self.nn_optimizer = optim.SGD(self.model.parameters(), lr=float(lr))

        # Device selection.
        if a_gpu :
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
           self.device = "cpu"

        #Move the model to respective device
        self.model.to(self.device)
        #Final Model after modification.
        print("Final Model generated")
        print(self.model)


    def create_model(self, hidden_units):
        """
        Builds the model based on
        model name and number of hidden layers parameters.
        """
        try:
           self.model = getattr(models, str(self.model_name))(pretrained=True)
           #print(self.model)
        except:
           raise ValueError('Invalid model name ' + self.model_name)

        for param in self.model.parameters():
           param.requires_grad = False

        if self.model_name == "resnet18":
           self.model.fc = self.classifier_network

        print(self.model_name)

        if self.model_name in { "vgg16",  "vgg19",  "densenet121", "alexnet"}:
           self.model.classifier = self.classifier_network
           print("Final Model Classifier")
           print(self.model.classifier)


        return self.model


    def train(self, input_t,labels_t, epoch, total, running_loss, accuracy):
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

        ps = torch.exp(output)

        # Calculate loss
        loss = self.nn_entropy(output, labels)

        # Backpropagation.
        loss.backward()
        # Accumulate the backpropagation gradients.
        self.nn_optimizer.step()

        # Training loss accumulation.
        running_loss += loss.item()
        # get the top predictions.
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return running_loss, accuracy


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
        # Running loss of validation.
        running_loss += loss.item()
        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return running_loss, accuracy


    def save_checkpoint(self,data_loader,path,lr,epochs):
        """
        Save a model checkpoint in a path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint_path = os.path.join(path,'checkpoint.pth')
        self.model.class_to_idx = data_loader.dataset.class_to_idx

        if ( self.model_name == "resnet18" ):
            checkpoint = {
                'model': self.model,
                'output_size': 102,
                'input_size': self.model_input_features,
                'state_dict': self.model.state_dict(),
                'fc' : self.classifier_network,
                'epochs':epochs,
                'arch':self.model_name,
                'batch_size':64,
                'learning_rate':lr,
                'class_to_idx': self.model.class_to_idx,
                'optimizer_dict': self.nn_optimizer.state_dict(),
                'batch_size': data_loader.batch_size,
              }

        elif  self.model_name in { "vgg16",  "vgg19",  "densenet121", "alexnet"}:
             checkpoint = {
                'model': self.model,
                'output_size': 102,
                'input_size': self.model_input_features,
                'state_dict': self.model.state_dict(),
                'classifier': self.classifier_network,
                'epochs':epochs,
                'arch':self.model_name,
                'batch_size':64,
                'learning_rate':lr,
                'class_to_idx': self.model.class_to_idx,
                'optimizer_dict': self.nn_optimizer.state_dict(),
                'batch_size': data_loader.batch_size,
              }

        print("Saving checkpoint")
        torch.save(checkpoint, checkpoint_path)
