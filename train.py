import argparse
from data_loaders_transform import get_data_loaders
from ClassifierNetwork import ClassifierNetwork


def get_cmdline_parameters():
    """
    Parsing command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", default=".", action="store")
    parser.add_argument("--save_dir", default=".", help="Checkpoint directory path")
    parser.add_argument("--arch", default="vgg19", help="Model architecture")
    parser.add_argument("--learning_rate", default=0.002, help="Learning rate", action="store", dest="lr")
    parser.add_argument("--hidden_units", default=4096, help="Number of hidden units")
    parser.add_argument("--epochs", default=24, help="Number of epochs")
    parser.add_argument("--gpu", default=False, help="Use GPU for training", action='store_true')

    return parser.parse_args()


def main():

    # Get all command line arguments.
    args = get_cmdline_parameters()

    # Load the image datasets.
    image_datasets, data_loaders = get_data_loaders(args.data_path)

    # Intialize the trainer
    classifier_net = ClassifierNetwork(args.gpu,args.lr,args.arch, args.hidden_units)



    # Running the iterations for training.
    for epoch in range(int(args.epochs)):

        #training_count
        pass_t = 0
        #validation_count
        pass_v = 0
        #Intialize accuracy
        accuracy_train = 0
        accuracy_valid = 0
        # Intialize the training loss and validation loss.
        running_loss_t = 0
        running_loss_v = 0

        # Training phase.
        for input_t,labels_t in data_loaders['train']:
            pass_t += 1
            #print(data_t)
            running_loss_t, accuracy_train = classifier_net.train(input_t,labels_t, epoch,args.epochs,running_loss_t,accuracy_train)
            print("Training_pass {}".format(pass_t))

        # Validation phase
        for input_v,labels_v in data_loaders['val']:
            pass_v += 1
            #print(data_v)
            running_loss_v, accuracy_valid = classifier_net.validate(input_v,labels_v, epoch,args.epochs,running_loss_v,accuracy_valid)
            print("Validation_pass {}".format(pass_v))

        # Statistics for the training and validation phase.
        print("\nTraining Epoch_Number: {}/{} ".format(epoch+1, args.epochs))
        print("\nTraining Loss: {:.4f}  ".format(running_loss_t/pass_t))
        print("\nTraining accuracy: {:.4f}".format(accuracy_train/pass_t))
        # Validation stats.
        print("\nValidation Epoch_Number: {}/{} ".format(epoch+1, args.epochs))
        print("\nValidation Loss: {:.4f}  ".format(running_loss_v/pass_v))
        print("\nValidation Accuracy: {:.4f}".format(accuracy_valid/pass_v))

    classifier_net.save_checkpoint(data_loaders['train'], args.save_dir, args.lr, args.epochs)


main()
