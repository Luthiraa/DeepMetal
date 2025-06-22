#!/usr/bin/env python3
# export_model.py - dynamic model exporter for multiple datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from archive.models import get_dataset_config, create_model, get_available_models

# ==================== Dataset Loaders ====================

class GTSRBDataset(Dataset):
    """custom dataset for gtsrb traffic signs"""
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        if train:
            self.data_dir = os.path.join(root_dir, 'Train')
            self.images = []
            self.labels = []
            
            # load training data from multiple class directories
            for class_id in range(43):  # gtsrb has 43 classes
                class_dir = os.path.join(self.data_dir, f'{class_id:05d}')
                if os.path.exists(class_dir):
                    csv_file = os.path.join(class_dir, f'GT-{class_id:05d}.csv')
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file, delimiter=';')
                        for _, row in df.iterrows():
                            img_path = os.path.join(class_dir, row['Filename'])
                            if os.path.exists(img_path):
                                self.images.append(img_path)
                                self.labels.append(row['ClassId'])
        else:
            # load test data
            self.data_dir = os.path.join(root_dir, 'Test')
            self.images = []
            self.labels = []
            
            csv_file = os.path.join(self.data_dir, 'GT-final_test.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, delimiter=';')
                for _, row in df.iterrows():
                    img_path = os.path.join(self.data_dir, row['Filename'])
                    if os.path.exists(img_path):
                        self.images.append(img_path)
                        self.labels.append(row['ClassId'])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((32, 32))  # resize to 32x32
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset_data(dataset_name, batch_size=64, num_workers=2, data_root='./data'):
    """dynamically loads dataset based on name"""
    config = get_dataset_config(dataset_name)
    
    # create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['normalization']['mean'], 
                        config['normalization']['std'])
    ])
    
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_root, 
            train=True,
            download=True, 
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform
        )
        
    elif dataset_name == 'gtsrb':
        gtsrb_root = os.path.join(data_root, 'GTSRB')
        
        # resize transform for gtsrb
        gtsrb_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(config['normalization']['mean'], 
                            config['normalization']['std'])
        ])
        
        train_dataset = GTSRBDataset(
            root_dir=gtsrb_root,
            train=True,
            transform=gtsrb_transform
        )
        
        test_dataset = GTSRBDataset(
            root_dir=gtsrb_root,
            train=False,
            transform=gtsrb_transform
        )
    
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    
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

# ==================== Training Functions ====================

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cuda', dataset_name='mnist'):
    """trains the model on specified dataset"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"training {dataset_name} model on device: {device}")
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

def export_model(model, model_name, dataset_name, output_dir='models'):
    """exports trained model in converter-compatible format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # set to evaluation mode
    model.eval()
    
    # save complete model
    model_path = os.path.join(output_dir, f'{dataset_name}_{model_name}.pth')
    torch.save(model, model_path, _use_new_zipfile_serialization=False)
    print(f"✅ exported complete model to: {model_path}")
    
    # save state dict only (alternative format)
    state_dict_path = os.path.join(output_dir, f'{dataset_name}_{model_name}_state_dict.pth')
    torch.save(model.state_dict(), state_dict_path)
    print(f"✅ exported state dict to: {state_dict_path}")
    
    # print model summary
    print(f"\n📋 model summary for {dataset_name}_{model_name}:")
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

def test_inference(model, test_loader, device='cuda', num_samples=5, dataset_name='mnist'):
    """tests model inference with sample predictions"""
    model.eval()
    model = model.to(device)
    
    print(f"\n🔍 testing {dataset_name} inference on {num_samples} samples:")
    
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

def create_test_converter_script(model_path, dataset_name):
    """creates a test script to run all converters"""
    script_content = f'''#!/bin/bash
# test_conversion_{dataset_name}.sh - tests all conversion approaches

echo "🔄 testing dynamic neural network converters"
echo "dataset: {dataset_name}"
echo "model: {model_path}"
echo

echo "📝 testing c converter..."
python converter.py {model_path}
echo

echo "✅ conversion complete!"
echo "check output/ directory for generated files"
'''
    
    script_name = f'test_conversion_{dataset_name}.sh'
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_name, 0o755)  # make executable
    print(f"📝 created test script: {script_name}")

# ==================== Main Function ====================
    """loads mnist dataset with preprocessing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
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

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cuda'):
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

def test_inference(model, test_loader, device='cuda', num_samples=5):
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
    parser = argparse.ArgumentParser(description='dynamic model exporter for multiple datasets')
    
    # get available models
    available_models = get_available_models()
    available_datasets = list(available_models.keys())
    available_model_types = list(next(iter(available_models.values())).keys())
    
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=available_datasets,
                        help=f'dataset to use: {available_datasets}')
    parser.add_argument('--model-type', type=str, default='hybrid',
                        choices=available_model_types,
                        help=f'type of model to create: {available_model_types}')
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
    parser.add_argument('--data-root', type=str, default='./data',
                        help='root directory for datasets')
    parser.add_argument('--use-sequential', action='store_true', default=True,
                        help='use sequential models (better for converter compatibility)')
    
    args = parser.parse_args()
    
    # determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 creating {args.dataset} {args.model_type} model for converter workflow")
    print(f"device: {device}")
    print(f"sequential model: {args.use_sequential}")
    print()
    
    # create model
    model = create_model(args.dataset, args.model_type, args.use_sequential)
    print(f"📋 model architecture:")
    print(model)
    print()
    
    if not args.no_train:
        # load data
        print(f"📥 loading {args.dataset} dataset...")
        try:
            train_loader, test_loader = load_dataset_data(
                args.dataset, 
                args.batch_size, 
                data_root=args.data_root
            )
            
            # train model
            print(f"🎯 training {args.dataset} model...")
            train_model(
                model, 
                train_loader, 
                test_loader, 
                args.epochs, 
                args.lr, 
                device, 
                args.dataset
            )
            
            # test inference
            test_inference(model, test_loader, device, dataset_name=args.dataset)
            
        except Exception as e:
            print(f"❌ error loading/training {args.dataset} dataset: {e}")
            print("💡 you can still export the model with --no-train")
            if not args.no_train:
                return
    else:
        print(f"⏭️ skipping training (--no-train specified)")
    
    # export model
    model_name = f'{args.model_type}_model'
    model_path = export_model(model, model_name, args.dataset, args.output_dir)
    
    # create test script
    create_test_converter_script(model_path, args.dataset)
    
    print(f"\n🎉 {args.dataset} model export complete!")
    print(f"to test converters, run: ./test_conversion_{args.dataset}.sh")
    print(f"or manually:")
    print(f"  python converter.py {model_path}")
    
    # show available combinations
    print(f"\n📚 available model combinations:")
    for dataset in available_datasets:
        for model_type in available_model_types:
            print(f"  --dataset {dataset} --model-type {model_type}")

if __name__ == '__main__':
    main()