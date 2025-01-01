import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine

# Read OSNet model
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

transform = transforms.Compose([
    transforms.Resize((256, 128)),  # Resize to 256x128
    transforms.ToTensor(),  # Conversion to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
])


def extract_descriptor(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0)
    features = extractor.model(image_tensor.to(extractor.device)).cpu().detach().numpy()
    return features.squeeze()


def compare_descriptors(desc1, desc2):
    distance = cosine(desc1, desc2)
    return distance
