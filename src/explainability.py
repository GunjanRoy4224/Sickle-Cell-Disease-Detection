import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.config import DEVICE, IMG_SIZE, SEED

def reshape_transform(tensor):
    bs, num_tokens, channels = tensor.shape
    side = int(math.sqrt(num_tokens))
    return tensor.transpose(1, 2).reshape(bs, channels, side, side)

def generate_grad_cam(model, test_df, val_tfms, model_name='swin_base'):
    model.eval()
    
    # Handle Architecture Showdown routing
    if 'swin' in model_name.lower():
        target_layer = model.layers[-2].blocks[-1].norm2
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        # Fallback for standard CNNs like ResNet
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])

    samples = test_df.sample(min(3, len(test_df)), random_state=SEED)
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax_row, row in zip(axes, samples.itertuples()):
        raw_img = Image.open(row.path).convert('RGB')
        
        # Center crop to remove microscope edges
        width, height = raw_img.size
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        raw_img = raw_img.crop((left, top, right, bottom))
        
        display_img = raw_img.resize((IMG_SIZE, IMG_SIZE))
        input_tensor = val_tfms(display_img).unsqueeze(0).to(DEVICE)

        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(
            input_tensor=input_tensor, 
            targets=targets, 
            aug_smooth=True, 
            eigen_smooth=True
        )[0]

        rgb_img = np.array(display_img).astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax_row[0].imshow(display_img)
        ax_row[0].set_title(f'Original ({row.class_name})')
        ax_row[0].axis('off')

        ax_row[1].imshow(visualization)
        ax_row[1].set_title('Grad-CAM (Smoothed)')
        ax_row[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'grad_cam_results_{model_name}.png')
    plt.close()