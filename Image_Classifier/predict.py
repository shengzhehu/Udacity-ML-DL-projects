import argparse
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image
from model import create_model
import json

def main():
    # Creates & retrieves Command Line Arguments
    parser = argparse.ArgumentParser(description='Train neural network - Image Classifier')
    parser.add_argument('input_image', action='store', type=str)
    parser.add_argument('checkpoint', action='store', type=str)
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
    parser.add_argument('--category_names ', action='store', dest='category_names', type=str)
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    input_image = args.input_image
    category_names = args.category_names
    top_k = args.top_k
    device = args.device
    
    # Loads model from checkpoint
    supermodel = load_checkpoint(checkpoint)
    
    # Sets criterion
    criterion = nn.NLLLoss()

    # Predicts output & prints results
    probs, classes = predict(image_path, supermodel)
    print(probs)
    print(classes)

def load_checkpoint(filepath):
    ''' Loads checkpoint from file
        returns model
    '''
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    model.epochs = checkpoint['epochs']
    model.arch = checkpoint['arch']
    model.state_dict = checkpoint['state_dict']
    model.criterion = checkpoint['criterion']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
   
    processed_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = processed_image(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    model.eval()
    model = model.cuda()
    with torch.no_grad():
        output = model.forward(processed_image.float().cuda().unsqueeze_(0))
    
    # Calculate the class probabilities (softmax) for image
    ps = F.softmax(output, dim=1)
    
    top = torch.topk(ps, topk)
    
    probs = top[0][0].cpu().numpy()
    classes = [idx_to_class[i] for i in top[1][0].cpu().numpy()]
    
    return probs, classes

# Call to main function to run the program
if __name__ == "__main__":
    main()