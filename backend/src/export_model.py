#!/usr/bin/env python3
# export_model.py - creates and exports pytorch models compatible with dynamic converters

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader

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

def create_sequential_model(model_type='linear'):
    """creates nn.Sequential model for better converter compatibility"""
    if model_type == 'linear':
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    elif model_type == 'conv':
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
            nn.Linear(128, 10)
        )
    elif model_type == 'hybrid':
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
            nn.Linear(32, 10)
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")

def load_mnist_data(batch_size=64, num_workers=2):
    """loads mnist dataset with preprocessing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mnist normalization
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    """trains the model on mnist dataset"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"training on device: {device}")
    print(f"model architecture:")
    print(model)
    print()
    
    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'epoch: {epoch+1}/{epochs}, batch: {batch_idx}, '
                      f'loss: {loss.item():.4f}, '
                      f'acc: {100.*train_correct/train_total:.2f}%')
        
        # validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        print(f'epoch {epoch+1}: train_acc={100.*train_correct/train_total:.2f}%, '
              f'test_acc={test_accuracy:.2f}%')
        print()

def export_model(model, model_name, output_dir='models'):
    """exports trained model in converter-compatible format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # set to evaluation mode
    model.eval()
    
    # save complete model
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save(model, model_path, _use_new_zipfile_serialization=False)
    print(f"✅ exported complete model to: {model_path}")
    
    # save state dict only (alternative format)
    state_dict_path = os.path.join(output_dir, f'{model_name}_state_dict.pth')
    torch.save(model.state_dict(), state_dict_path)
    print(f"✅ exported state dict to: {state_dict_path}")
    
    # print model summary
    print(f"\n📋 model summary for {model_name}:")
    print(f"   layers: {len(list(model.modules()))-1}")  # -1 for root module
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   total parameters: {total_params:,}")
    print(f"   trainable parameters: {trainable_params:,}")
    
    # layer breakdown
    layer_types = {}
    for module in model.modules():
        layer_type = type(module).__name__
        if layer_type != 'Sequential' and layer_type != type(model).__name__:
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    print(f"   layer breakdown:")
    for layer_type, count in layer_types.items():
        print(f"     {layer_type}: {count}")
    
    return model_path

def test_inference(model, test_loader, device='cpu', num_samples=5):
    """tests model inference with sample predictions"""
    model.eval()
    model = model.to(device)
    
    print(f"\n🔍 testing inference on {num_samples} samples:")
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data[:1])  # process single sample
            
            probabilities = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1)
            
            print(f"sample {i+1}: true={target[0].item()}, "
                  f"predicted={predicted[0].item()}, "
                  f"confidence={probabilities[0][predicted[0]].item():.4f}")

def create_test_converter_script(model_path):
    """creates a test script to run all three converters"""
    script_content = f'''#!/bin/bash
# test_conversion.sh - tests all three conversion approaches

echo "🔄 testing dynamic neural network converters"
echo "model: {model_path}"
echo

echo "📝 testing c converter..."
python converter.py {model_path}
echo

echo "🏗️ testing llvm converter..."
python llvm.py {model_path}
echo

echo "🎯 testing c++ template converter..."
python pytoc.py {model_path}
echo

echo "✅ all conversions complete!"
echo "check output/ directory for generated files"
'''
    
    with open('test_conversion.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('test_conversion.sh', 0o755)  # make executable
    print(f"📝 created test script: test_conversion.sh")

def main():
    parser = argparse.ArgumentParser(description='export pytorch model for converter workflow')
    parser.add_argument('--model-type', type=str, default='hybrid',
                        choices=['linear', 'conv', 'hybrid'],
                        help='type of model to create')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--no-train', action='store_true',
                        help='skip training, just create and export model')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='output directory for exported models')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='device to use for training')
    
    args = parser.parse_args()
    
    # determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 creating {args.model_type} model for converter workflow")
    print(f"device: {device}")
    print()
    
    # create model
    model = create_sequential_model(args.model_type)
    print(f"📋 model architecture:")
    print(model)
    print()
    
    if not args.no_train:
        # load data
        print("📥 loading mnist dataset...")
        train_loader, test_loader = load_mnist_data(args.batch_size)
        
        # train model
        print("🎯 training model...")
        train_model(model, train_loader, test_loader, args.epochs, args.lr, device)
        
        # test inference
        test_inference(model, test_loader, device)
    else:
        print("⏭️ skipping training (--no-train specified)")
    
    # export model
    model_name = f'mnist_{args.model_type}_model'
    model_path = export_model(model, model_name, args.output_dir)
    
    # create test script
    create_test_converter_script(model_path)
    
    print(f"\n🎉 model export complete!")
    print(f"to test converters, run: ./test_conversion.sh")
    print(f"or manually:")
    print(f"  python converter.py {model_path}")
    print(f"  python llvm.py {model_path}")  
    print(f"  python pytoc.py {model_path}")

if __name__ == '__main__':
    main()