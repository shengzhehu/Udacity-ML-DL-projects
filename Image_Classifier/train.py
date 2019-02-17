import argparse
import numpy as np
import torch
import os
from torch import optim, nn
from torchvision import datasets, transforms
from PIL import Image
from model import create_model



def main():
    # Creates & retrieves Command Line Arguments
    parser = argparse.ArgumentParser(description='Image Classifier: Train')
    parser.add_argument('data_directory', action='store', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_directory', default=None, type=str)
    parser.add_argument('--arch', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], action='store', dest='arch', default='vgg11', type=str)
    parser.add_argument('--learning_rate', action='store', dest='lr', default=0.001, type=float)
    parser.add_argument('--epochs', action='store', dest='epochs', default=3, type=int)
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu')
    args = parser.parse_args()
    data_directory = args.data_directory
    arch = args.arch
    lr = args.lr
    device = args.device
    save_directory = args.save_directory
    epochs = args.epochs
    
    # Creates transforms
    transforms = create_transforms()
    
    # Loads image datasets
    image_datasets = create_image_datasets(data_directory, transforms)
    
    # Creates dataloaders
    dataloaders = create_dataloaders(image_datasets)
    
    # Creates NN model
    model = create_model(arch)
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Sets criterion
    criterion = nn.NLLLoss()
    
    # Sets optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # Trains NN
    train_model(model, dataloaders['train'], dataloaders['valid'], epochs, 40, criterion, optimizer, device)
   
    # Checks accuracy (test data)
    check_accuracy_on_test(model, dataloaders['test'], device)
    
    if save_directory:
        checkpoint = {
            'epochs': epochs,
            'arch': arch,
            'state_dict': model.state_dict(),
            'classifier': model.classifier,
            'optimizer_dict': optimizer.state_dict(),
            'class_to_idx': image_datasets['train'].class_to_idx,
            'criterion': criterion
        }
        torch.save(checkpoint, save_dir + '/checkpoint.pth')
        print('Model saved successfully!')

def create_transforms():
    """ Returns transforms for the training, validation, and testing sets
    """
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # Test
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # validate
    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    data_transforms = {'train' : train_transforms, 
                       'test' : test_transforms,
                       'valid' : valid_transforms}
    return data_transforms
    
def create_image_datasets(data_dir, transforms):
    """ Returns image datasets, with transform applied
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Loads the datasets
    train_data = datasets.ImageFolder(train_dir, transform=transforms['train'])
    test_data = datasets.ImageFolder(test_dir, transform=transforms['test'])
    valid_data = datasets.ImageFolder(valid_dir, transform=transforms['valid'])
    
    image_datasets = {'train' : train_data,
                      'test' : test_data,
                      'valid' : valid_data}
    return image_datasets

def create_dataloaders(image_datasets):
    """ Returns dataloaders
    """
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    dataloaders = {'train' : train_loader,
                   'test' : test_loader,
                   'valid' : valid_loader}
    
    return dataloaders

def check_accuracy_on_test(model, testloader, device='cpu'):
    """ Test accuracy of the model
    """
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def train_model(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device='cpu'):
    """ Trains the NN
    """

    epochs = 5
    print_every = 1000
    steps = 0
    lr = 0.001
    running_loss = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device);    
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
                
# Call to main function to run the program
if __name__ == "__main__":
    main()