# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:42:15 2024

@author: 19714
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from old_dataset import MPISintelDataset  # make sure to implement this or import your dataset class
from model import PWCNetMultiScale  # make sure to implement this or import your model class
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os

def count_png_files_in_final_pass(dataset_root):
    final_pass_dir = os.path.join(dataset_root, 'training', 'final')
    total_png_files = 0

    for root, dirs, files in os.walk(final_pass_dir):
        for file in files:
            if file.endswith('.png'):
                total_png_files += 1

    return total_png_files

class MultiScaleEPELoss(nn.Module):
    def __init__(self, alpha, epsilon, q):
        super(MultiScaleEPELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = q

    def forward(self, predicted_flows, ground_truth_flows):
        loss = 0.0
        for l, (pred_flow, gt_flow) in enumerate(zip(predicted_flows, ground_truth_flows)):
            # Upsample predicted flow to match the size of the ground truth flow
            if pred_flow.size()[-2:] != gt_flow.size()[-2:]:
                pred_flow = F.interpolate(pred_flow, size=gt_flow.size()[-2:], mode='bilinear', align_corners=False)
                # Scale the flow values to match the upsampling
                scale_factor = gt_flow.size()[-1] / pred_flow.size()[-1]
                pred_flow *= scale_factor

            # Compute the endpoint error (EPE)
            epe = torch.norm(pred_flow - gt_flow + self.epsilon, p=self.q, dim=1)
            layer_loss = self.alpha[l] * epe.mean()
            loss += layer_loss

        return loss

# Usage
#dataset_root = "C:/Users/19714/OneDrive - Oregon State University/Assignments/Winter 2024/Deep learning/Final Project/MPI-Sintel-complete"
#total_png_files = count_png_files_in_final_pass(dataset_root)
#print(f"Total PNG files in the final pass: {total_png_files}")

# Configuration
feature_channels = 64
max_displacement = 4
num_epochs = 30
batch_size = 4
learning_rate = 1e-4
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters for the loss function - these will need to be defined based on your application
alpha = [0.32, 0.08, 0.02, 0.01, 0.005]  # Example values
epsilon = 0.001  # Example value
q = 0.4  # Example value

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and flow maps to 256x256
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    # Add any additional transformations here
])

# Load the dataset
dataset_root = "Project/training/Sintel"
train_dataset = MPISintelDataset(root_dir=dataset_root, split='training', pass_type='clean', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Check the size of the training dataset and the number of batches
# print(f"Size of training dataset: {len(train_dataset)}")
# print(f"Number of batches in training loader: {len(train_loader)}")

#%%
# Initialize the model
model = PWCNetMultiScale(feature_channels=feature_channels, max_displacement=max_displacement).to(device)

# Instantiate the loss function
loss_function = MultiScaleEPELoss(alpha, epsilon, q).to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()  # set the model to training mode
    total_loss = 0
    loss_count = 0  # Initialize a counter for the number of loss values
    for i, (img1, img2, flow_gt) in enumerate(train_loader):
        img1, img2, flow_gt = img1.to(device), img2.to(device), flow_gt.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        flow_pred = model(img1, img2)

        # Ensure flow_pred is a list
        if not isinstance(flow_pred, list):
            flow_pred = [flow_pred]

        # Compute the loss
        loss = loss_function(flow_pred, flow_gt)  # flow_gt is kept as a tensor
        total_loss += loss.item()
        loss_count += 1  # Increment the loss counter

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print log info
        if i % log_interval == 0:
            print(f'Epoch {epoch}/{num_epochs} [{i}/{len(train_loader)}], Loss: {loss.item():.6f}')

    # Print epoch info
    avg_loss = total_loss / len(train_loader)
    print(f'End of Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    print(f'Number of loss values: {loss_count}, Number of batches: {len(train_loader)}')  # Check the number of loss values
    # Update the learning rate
    scheduler.step()


# Save the model after training
torch.save(model.state_dict(), 'pwc_net.pth')

#%%
# test_path = 'C:/Users/19714/OneDrive - Oregon State University/Assignments/Winter 2024/Deep learning/Final Project/MPI-Sintel-complete/train/frame_0001.png'
# try:
#     with open(test_path, 'r') as f:
#         print("File is accessible")
# except Exception as e:
#     print("Error accessing file:", e)
