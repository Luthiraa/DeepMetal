#!/usr/bin/env python3
# models.py - pytorch model definitions for deepmetal conversion

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== MNIST Models ====================

class MNISTLinearModel(nn.Module):
    """simple fully connected model for mnist classification"""
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super(MNISTLinearModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # output layer (no activation - will be handled in softmax)
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # flatten input for linear layers
        x = x.view(x.size(0), -1)
        return self.network(x)

class MNISTConvModel(nn.Module):
    """convolutional model for mnist classification"""
    def __init__(self, num_classes=10):
        super(MNISTConvModel, self).__init__()
        
        self.features = nn.Sequential(
            # first conv block: 1x28x28 -> 16x26x26
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # second conv block: 16x26x26 -> 32x24x24  
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # third conv block: 32x24x24 -> 32x12x12
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # calculate flattened size after conv layers
        # input: 1x28x28
        # after conv1: 16x26x26 (28-3+1=26)
        # after conv2: 32x24x24 (26-3+1=24)  
        # after conv3: 32x12x12 ((24+2-3)/2+1=12)
        conv_output_size = 32 * 12 * 12
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

class MNISTHybridModel(nn.Module):
    """hybrid model with both conv and linear layers"""
    def __init__(self, num_classes=10):
        super(MNISTHybridModel, self).__init__()
        
        # define layers individually for better converter compatibility
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 1x28x28 -> 8x28x28
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 8x28x28 -> 16x14x14
        self.relu2 = nn.ReLU()
        
        # flattened size: 16 * 14 * 14 = 3136
        self.linear1 = nn.Linear(16 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        
        self.linear2 = nn.Linear(128, 32)
        self.relu4 = nn.ReLU()
        
        self.linear3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # conv layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # flatten for linear layers
        x = x.view(x.size(0), -1)
        
        # linear layers
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        
        return x

# ==================== GTSRB Models ====================

class GTSRBLinearModel(nn.Module):
    """simple fully connected model for gtsrb traffic sign classification"""
    def __init__(self, input_size=32*32*3, hidden_sizes=[256, 128], num_classes=43):
        super(GTSRBLinearModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # flatten input for linear layers
        x = x.view(x.size(0), -1)
        return self.network(x)

class GTSRBConvModel(nn.Module):
    """convolutional model for gtsrb traffic sign classification"""
    def __init__(self, num_classes=43):
        super(GTSRBConvModel, self).__init__()
        
        self.features = nn.Sequential(
            # first conv block: 3x32x32 -> 16x30x30
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # second conv block: 16x30x30 -> 32x28x28
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # third conv block: 32x28x28 -> 64x14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # fourth conv block: 64x14x14 -> 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # calculate flattened size after conv layers
        conv_output_size = 64 * 7 * 7
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

class GTSRBHybridModel(nn.Module):
    """hybrid model with both conv and linear layers for gtsrb"""
    def __init__(self, num_classes=43):
        super(GTSRBHybridModel, self).__init__()
        
        # conv layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 3x32x32 -> 16x32x32
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16x32x32 -> 32x16x16
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x16x16 -> 64x8x8
        self.relu3 = nn.ReLU()
        
        # flattened size: 64 * 8 * 8 = 4096
        self.linear1 = nn.Linear(64 * 8 * 8, 256)
        self.relu4 = nn.ReLU()
        
        self.linear2 = nn.Linear(256, 64)
        self.relu5 = nn.ReLU()
        
        self.linear3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # conv layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        # flatten for linear layers
        x = x.view(x.size(0), -1)
        
        # linear layers
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = self.relu5(x)
        x = self.linear3(x)
        
        return x

# ==================== Dynamic Model Factory ====================

def get_dataset_config(dataset_name):
    """returns configuration for different datasets"""
    configs = {
        'mnist': {
            'input_channels': 1,
            'input_size': 28,
            'num_classes': 10,
            'linear_input_size': 784,  # 28*28*1
            'normalization': {'mean': (0.1307,), 'std': (0.3081,)}
        },
        'gtsrb': {
            'input_channels': 3,
            'input_size': 32,
            'num_classes': 43,
            'linear_input_size': 3072,  # 32*32*3
            'normalization': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"unsupported dataset: {dataset_name}. supported: {list(configs.keys())}")
    
    return configs[dataset_name]

def create_sequential_model(dataset_name, model_type='linear'):
    """creates nn.Sequential model for any dataset dynamically"""
    config = get_dataset_config(dataset_name)
    
    if model_type == 'linear':
        if dataset_name == 'mnist':
            return nn.Sequential(
                nn.Linear(config['linear_input_size'], 128),
                nn.ReLU(),
                nn.Linear(128, 64), 
                nn.ReLU(),
                nn.Linear(64, config['num_classes'])
            )
        elif dataset_name == 'gtsrb':
            return nn.Sequential(
                nn.Linear(config['linear_input_size'], 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, config['num_classes'])
            )
    
    elif model_type == 'conv':
        if dataset_name == 'mnist':
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),  # 1x28x28 -> 16x26x26
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # 16x26x26 -> 32x24x24
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 32x24x24 -> 32x12x12
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 12 * 12, 128),
                nn.ReLU(),
                nn.Linear(128, config['num_classes'])
            )
        elif dataset_name == 'gtsrb':
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),  # 3x32x32 -> 16x30x30
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # 16x30x30 -> 32x28x28
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x28x28 -> 64x14x14
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 64x14x14 -> 64x7x7
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, config['num_classes'])
            )
    
    elif model_type == 'hybrid':
        if dataset_name == 'mnist':
            return nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),   # 1x28x28 -> 8x28x28
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x28x28 -> 16x14x14
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 14 * 14, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(), 
                nn.Linear(32, config['num_classes'])
            )
        elif dataset_name == 'gtsrb':
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3x32x32 -> 16x32x32
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x32x32 -> 32x16x16
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x16x16 -> 64x8x8
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, config['num_classes'])
            )
    
    else:
        raise ValueError(f"unknown model type: {model_type}")

def create_custom_model(dataset_name, model_type='linear'):
    """creates custom model classes for any dataset"""
    if dataset_name == 'mnist':
        if model_type == 'linear':
            return MNISTLinearModel()
        elif model_type == 'conv':
            return MNISTConvModel()
        elif model_type == 'hybrid':
            return MNISTHybridModel()
    elif dataset_name == 'gtsrb':
        if model_type == 'linear':
            return GTSRBLinearModel()
        elif model_type == 'conv':
            return GTSRBConvModel()
        elif model_type == 'hybrid':
            return GTSRBHybridModel()
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")

# ==================== Model Registry ====================

MODEL_REGISTRY = {
    'mnist': {
        'linear': MNISTLinearModel,
        'conv': MNISTConvModel,
        'hybrid': MNISTHybridModel
    },
    'gtsrb': {
        'linear': GTSRBLinearModel,
        'conv': GTSRBConvModel,
        'hybrid': GTSRBHybridModel
    }
}

def get_available_models():
    """returns all available model combinations"""
    return MODEL_REGISTRY

def create_model(dataset_name, model_type='linear', use_sequential=True):
    """unified model creation function"""
    if use_sequential:
        return create_sequential_model(dataset_name, model_type)
    else:
        return create_custom_model(dataset_name, model_type)
