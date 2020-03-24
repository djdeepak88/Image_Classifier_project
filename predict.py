import argparse
import json
import numpy as np
import torch
from torch import optim
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
import json


def get_cmdline_parameters():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('input')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', default=5, help='Number of top K most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Category names file path')
    parser.add_argument('--gpu', dest='gpu', default=False, help='Use GPU for inference', action='store_true')

    return parser.parse_args()


def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def predict(image_path, model, device, topk):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    print(device)
    # if the device is GPU or CPU.
    if str(device) == "cuda":
        # Move model parameters to the GPU
        print("Number of GPUs: {}".format(torch.cuda.device_count()))
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))

    model.to(device)
    # turn off dropout
    model.eval()
    # The image
    image = process_image(image_path)
    # Tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    # Transfer the images to device.
    image = image.to(device)
    # pass the images in forward pass.
    output = model.forward(image)
    # Get the probabliites by taking the log of the data.
    probabilities = torch.exp(output).data
    # Getting probability and index.
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    print(prob)
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    print(index)

    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    #print(ind)

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])
    print(label)

    return prob, label


def load_checkpoint(checkpoint_path):
    """
    Loading the checkpoint into a model.
    """

    checkpoint = torch.load(checkpoint_path)
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.classifier.parameters(), float(learning_rate))
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return model, optimizer

def display_predictions(img_path,cat_to_name,prob,classes):

    max_index = np.argmax(prob)
    #print(max_index)
    max_probability = prob[max_index]
    #print(max_probability)
    label = classes[max_index]
    #print(label)

    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot2grid((15,10), (0,0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15,10), (9,2), colspan=5, rowspan=5)

    image = Image.open(img_path)
    ax1.axis('off')
    ax1.set_title(cat_to_name[label])
    ax1.imshow(image)

    top_labels = []
    for cl in classes:
        top_labels.append(cat_to_name[cl])

    y_pos = np.arange(5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_labels)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

    plt.show()

def main():

    # get commandline parameters.
    args = get_cmdline_parameters()

    # Load the category_names file.
    with open(args.category_names, 'r') as f:
        category_mappings = json.load(f)

    # Get the model and optimizer loaded from the checkpoint
    model, optimizer = load_checkpoint(args.checkpoint)

    # Device selection.
    if args.gpu :
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
       device = "cpu"

    # print(device)
    probabilities, classes = predict(args.input, model, device, int(args.top_k))
    # printing all the probabliites and categories.
    for probability, class_index in zip(probabilities, classes):
        print("{}: {}".format(category_mappings[class_index], probability))

    # Visualization of the top 5 categories.
    display_predictions(args.input, category_mappings, probabilities, classes)


main()
