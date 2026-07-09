import timm
from src.config import DEVICE

def get_model(model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=1):
    """
    Easily swap between architectures. 
    Examples: 'swin_base_patch4_window7_224', 'resnet50', 'efficientnet_b0'
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)