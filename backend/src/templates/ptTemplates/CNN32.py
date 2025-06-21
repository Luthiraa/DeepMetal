import torch
import torch.nn as nn

class CNN32(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN32, self).__init__()
        self.features = nn.Sequential(
            # Input: 1×32×32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # →16×32×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # →16×16×16

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # →32×16×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # →32×8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # → (32*8*8)=2048
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate & save
model = CNN32(num_classes=10)
torch.save(model.state_dict(), 'cnn32.pth')
print("Saved 32×32 grayscale CNN weights to cnn32.pth")