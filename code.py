import torch
import cv2
import numpy as np
from dagf import make_model

class DummyArgs:
    pass

args = DummyArgs()

model = make_model(args)

model_path = '/pre_trained/net_nearest_x16.pth'
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['state'])
model.eval()

depth_image_path = '/depth_image.jpg'
depth_image_lr = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

# Load the HR RGB image
rgb_image_path = '/rgb_image.jpg'
rgb_image_hr = cv2.imread(rgb_image_path)

# Resize the depth image to match the RGB image's resolution, if necessary
depth_image_lr_resized = cv2.resize(depth_image_lr, (rgb_image_hr.shape[1], rgb_image_hr.shape[0]))

# Convert the depth image to a PyTorch tensor
depth_tensor_lr = torch.tensor(depth_image_lr_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

with torch.no_grad():
    depth_tensor_hr = model(depth_tensor_lr)

# Convert the tensor to a NumPy array
depth_image_hr = depth_tensor_hr.squeeze().cpu().numpy() * 255.0

cv2.imwrite('upscaled_depth_image.jpg', depth_image_hr)

# Optionally, display the images
cv2.imshow('Upscaled Depth Image', depth_image_hr)
cv2.imshow('HR RGB Image', rgb_image_hr)
cv2.waitKey(0)
cv2.destroyAllWindows()
