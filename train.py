import torch.optim as optim
from dataset import AIDataset
from model import Net
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from configparser import ConfigParser
from tqdm import tqdm
from utils import setup_exp_folder, plot_line, create_class_report
import numpy as np


training_losses = []
testing_losses = []

training_matches_epochs = []
testing_matches_epochs = []

nr_of_epochs = 10

# Getting all the transforms used 
def get_transform():

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomRotation((0, 90)),
        transforms.RandomResizedCrop((600, 600)),
    ])

    return transform

# Function to get the loss and accuracy - used for testing against test_loader
def get_testing_loss_acc(loader):

    model.eval()

    test_loss = 0.0
    test_matches = 0

    for data in tqdm(loader):
        inputs, labels = data
              
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_matches += get_nr_matches(labels, outputs)
        test_loss += loss.item()
        
    return (test_loss, test_matches)

# Function to get the number of matches between the targets and outputs
def get_nr_matches(targets, outputs):

    outputs = torch.argmax(outputs, dim = 1)
    matches = torch.eq(targets, outputs)
    nr_of_mathces = matches.sum().item()

    return nr_of_mathces


# Load the dataset and split it into train and test
dataset = AIDataset("dataset_csv/all.csv", get_transform())
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

print (f"Length of train set: {train_set.__len__()}")
print (f"Length of test set: {test_set.__len__()}")

# Load the train and test loaders
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

# Initialize the model, criterion and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Training the model
for epoch in tqdm(range(nr_of_epochs)):

    print (f"Starting training for epoch: {epoch + 1}")
    model.train()

    running_train_loss = 0.0
    train_matches = 0

    for data in tqdm(train_loader):
        
        inputs, labels = data
        
        optimizer.zero_grad()
       
        outputs = model(inputs)

        train_matches += get_nr_matches(labels, outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()


    training_losses.append(running_train_loss / len(train_loader))
    training_matches_epochs.append(train_matches / train_set.__len__())

    test_loss, test_matches = get_testing_loss_acc(test_loader)
    testing_losses.append(test_loss / len(test_loader))
    testing_matches_epochs.append(test_matches / test_set.__len__())


results_path = setup_exp_folder()

print (f"Results path: {results_path}")

x = [x for x in range(1, len(training_losses) + 1)]

print (f"Training Loss: {training_losses}")
print (f"Testing Loss: {testing_losses}")

print (f"Training Matches: {training_matches_epochs}")
print (f"Testing Matches: {testing_matches_epochs}")

plot_line(x, np.array([training_losses, testing_losses]).T,
            "losses", "Losses over epochs", results_path, legends=["Training Loss", "Testing Loss"])

plot_line(x, np.array([training_matches_epochs, testing_matches_epochs]).T,
            "matches", "Matches over epochs", results_path, legends=["Training Accuracy", "Testing Accuracy"])

torch.save(model.state_dict(), f"{results_path}/model.pt")

print('Finished Training')


