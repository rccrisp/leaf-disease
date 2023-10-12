import torch
import torch.nn as nn

class diseaseClassifier(nn.Module):
    def __init__(self, base_model, num_channels, pretrained=True):
        super(diseaseClassifier, self).__init__()

        # Use the provided base model
        self.base_model = base_model(pretrained=pretrained)

        # Disable gradient computation for the base model's parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Custom toplayer architecture
        self.input_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),  # Global Average Pooling 2D
                                            nn.Dropout(0.2),
                                            nn.Linear(in_features=num_channels, out_features=128),  # Specify your_input_features
                                            nn.ReLU()
                                        )

        self.custom_input = nn.Linear(in_features=your_input_features, out_features=your_hidden_units)
        
        # Define a custom output layer
        self.custom_output = nn.Linear(in_features=your_hidden_units, out_features=num_classes)

    def forward(self, x):
        # Forward pass through the custom input layer
        x = self.custom_input(x)
        
        # Forward pass through the base model
        x = self.base_model(x)
        
        # Global average pooling
        x = torch.mean(x, [2, 3])  # Assuming 2D input (e.g., images)
        
        # Forward pass through the custom output layer
        x = self.custom_output(x)
        
        return x
