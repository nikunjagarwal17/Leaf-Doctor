
# Use in app.py: from marked_predictor_template import predict_image_with_markup

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def predict_image_with_markup(image_path, model, device, class_names):
    '''Predict image and return marked version with confidence'''
    img = Image.open(image_path).resize((224, 224))
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    if len(img_arr.shape) == 2:
        img_arr = np.stack([img_arr]*3, axis=-1)
    
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    pred_idx = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0].max().item()
    class_name = class_names[pred_idx]
    
    # Create marked image
    marked_img = Image.open(image_path).resize((224, 224))
    draw = ImageDraw.Draw(marked_img)
    
    text = f"{class_name} ({confidence*100:.1f}%)"
    draw.text((10, 10), text, fill=(255, 255, 0))
    
    return {
        'class': class_name,
        'confidence': float(confidence),
        'marked_image': marked_img
    }
