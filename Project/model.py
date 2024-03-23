# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:21:28 2024

@author: 19714
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class OpticalFlowEstimator(nn.Module):
    def __init__(self, feature_channels=64, max_displacement=4):
        super(OpticalFlowEstimator, self).__init__()
        self.max_displacement = max_displacement
        cost_volume_channels = self.calculate_cost_volume_channels()

        # Total input channels = feature channels + cost volume channels
        total_input_channels = feature_channels + cost_volume_channels

        self.conv1 = nn.Conv2d(total_input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def calculate_cost_volume_channels(self):
        # The cost volume will have a channel for each displacement in a 2D grid
        return (self.max_displacement * 2 + 1) ** 2

    def forward(self, f1, f2_warped, flow_up=None):
        # Compute the cost volume
        cost_volume = self.compute_cost_volume(f1, f2_warped, self.max_displacement)

        # Concatenate the cost volume with the feature map (f1)
        x = torch.cat([f1, cost_volume], dim=1)

        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flow = self.conv4(x)

        return flow

    def compute_cost_volume(self, f1, f2_warped, max_displacement):
        B, C, H, W = f1.size()
        cost_volume = torch.zeros(B, (max_displacement * 2 + 1) ** 2, H, W, device=f1.device)

        for i in range(-max_displacement, max_displacement + 1):
            for j in range(-max_displacement, max_displacement + 1):
                # Shift f2_warped
                shifted_f2_warped = torch.roll(f2_warped, shifts=(i, j), dims=(2, 3))

                # Handling borders by adding zero padding
                if i > 0:
                    shifted_f2_warped[:, :, :i, :] = 0
                elif i < 0:
                    shifted_f2_warped[:, :, i:, :] = 0
                if j > 0:
                    shifted_f2_warped[:, :, :, :j] = 0
                elif j < 0:
                    shifted_f2_warped[:, :, :, j:] = 0

                # Calculate the correlation (normalized cross-correlation)
                correlation = (f1 * shifted_f2_warped).sum(1).unsqueeze(1)
                correlation /= C  # Normalizing by the number of channels

                # Add to cost volume
                index = (i + max_displacement) * (2 * max_displacement + 1) + (j + max_displacement)
                cost_volume[:, index, :, :] = correlation.squeeze(1)

        return cost_volume

class ContextNetwork(nn.Module):
    def __init__(self):
        super(ContextNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(128, 96, kernel_size=3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(96, 64, kernel_size=3, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1)
        self.conv7 = nn.Conv2d(32, 2, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        flow = self.conv7(x)
        return flow

# # Example usage
# context_network = ContextNetwork()
# flow = torch.randn(1, 2, 128, 128)
# refined_flow = context_network(flow)
# print(refined_flow.shape)  # Should be (1, 2, 128, 128)

    
class MultiScaleEPELoss(nn.Module):
    def __init__(self, alpha, epsilon, q, gamma, start_level, num_levels):
        super(MultiScaleEPELoss, self).__init__()
        self.alpha = alpha  # This can be a list of weights for each level
        self.epsilon = epsilon
        self.q = q
        self.gamma = gamma
        self.start_level = start_level
        self.num_levels = num_levels

    def forward(self, predicted_flows, target_flow):
        loss = 0.0
        regularization_term = 0.0

        # We assume predicted_flows is a list of flow predictions from coarse to fine
        for l, flow_pred in enumerate(predicted_flows[self.start_level:], start=self.start_level):
            # Upsample predicted flow to match the size of the target flow if necessary
            if flow_pred.size()[-2:] != target_flow.size()[-2:]:
                scale_factor = target_flow.size()[-2] / flow_pred.size()[-2]
                flow_pred = F.interpolate(flow_pred, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                flow_pred *= scale_factor

            # Calculate the robust loss at the current scale
            epe = torch.norm(flow_pred - target_flow, dim=1)  # Endpoint error
            charbonnier_loss = torch.mean(self.alpha[l] * (epe + self.epsilon) ** self.q)
            loss += charbonnier_loss

        # Calculate the regularization term (weight decay)
        for param in self.parameters():
            regularization_term += param.norm(2)

        total_loss = loss + self.gamma * regularization_term
        return total_loss
    
    
    
class PWCNetMultiScale(nn.Module):
    def __init__(self, num_levels=3, feature_channels=64, max_displacement=4):
        super(PWCNetMultiScale, self).__init__()
        self.num_levels = num_levels
        self.feature_extractor = FeatureExtractor()
        self.flow_estimators = nn.ModuleList([
            OpticalFlowEstimator(feature_channels, max_displacement) for _ in range(num_levels)
        ])
        self.context_network = ContextNetwork()  # Add the context network

    def forward(self, img1, img2):
        # Create image pyramids
        img1_pyramid = [img1]
        img2_pyramid = [img2]
        for _ in range(1, self.num_levels):
            img1_pyramid.append(F.interpolate(img1_pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False))
            img2_pyramid.append(F.interpolate(img2_pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False))

        # Initialize flow
        flow = None
        predicted_flows = []  # Initialize a list to hold all scale predictions
        # Coarse-to-fine approach
        for level in reversed(range(self.num_levels)):
            # Extract features
            f1 = self.feature_extractor(img1_pyramid[level])
            f2 = self.feature_extractor(img2_pyramid[level])

            # Warp features from img2 to img1 using the upsampled flow from the previous level
            if flow is not None:
                flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
                f2_warped = F.grid_sample(f2, self.flow_to_grid(flow_up), mode='bilinear', padding_mode='border')
                flow = self.flow_estimators[level](f1, f2_warped, flow_up)
            else:
                f2_warped = f2
                flow = self.flow_estimators[level](f1, f2_warped)
            # Only append the intermediate predictions to the list, not the final one
            if level != 0:  # Assuming level 0 is the finest scale
                predicted_flows.append(flow)
            
        # Refine the final flow estimate using the context network
        refined_flow = self.context_network(flow)
        
        # Save the refined flow as well
        predicted_flows.append(refined_flow)

        # # Reverse the list so that the finest scale flow is first
        # predicted_flows = predicted_flows[::-1]

        return predicted_flows

    def flow_to_grid(self, flow):
        n, c, h, w = flow.size()
        grid_x, grid_y = torch.meshgrid(torch.arange(w, device=flow.device), torch.arange(h, device=flow.device))
        grid_x = grid_x.float().unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
        grid_y = grid_y.float().unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
        grid = torch.cat((grid_x, grid_y), dim=1)
        grid[:, 0, :, :] = (grid[:, 0, :, :] + flow[:, 0, :, :]) * 2 / (w - 1) - 1
        grid[:, 1, :, :] = (grid[:, 1, :, :] + flow[:, 1, :, :]) * 2 / (h - 1) - 1
        return grid.permute(0, 2, 3, 1)

# You can create an instance of the model like this:
# pwc_net_multi_scale = PWCNetMultiScale(num_levels=3)
