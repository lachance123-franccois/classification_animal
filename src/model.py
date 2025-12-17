from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

def get_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
