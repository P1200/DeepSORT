import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine

# 1. Wczytaj model (OSNet)
extractor = FeatureExtractor(
    model_name='osnet_x1_0',  # Model OSNet dla reidentyfikacji osób
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Użyj GPU jeśli dostępne
)

# 2. Transformacje obrazu
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # Przeskalowanie do wymiaru 256x128
    transforms.ToTensor(),  # Konwersja do tensora
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizacja
])


# 3. Funkcja do ekstrakcji deskryptora wyglądu
def extract_descriptor(image):
    """
    Pobiera deskryptor cech z obrazu.
    - image: Obiekt PIL.Image lub NumPy array.
    """
    if isinstance(image, np.ndarray):  # Jeśli obraz jest w formacie NumPy
        image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0)  # Transformacja i dodanie wymiaru batch
    features = extractor.model(image_tensor.to(extractor.device)).cpu().detach().numpy()
    return features.squeeze()


# 4. Porównanie dwóch deskryptorów
def compare_descriptors(desc1, desc2, threshold=0.5):
    """
    Porównuje dwa deskryptory cech i zwraca informację, czy to ta sama osoba.
    """
    distance = cosine(desc1, desc2)
    return distance
    # return distance < threshold, distance


# # 5. Przykład użycia
# # Wczytaj dwa wycinki obrazka
# image1 = Image.open("path_to_crop1.jpg")  # Wycinek obrazu 1
# image2 = Image.open("path_to_crop2.jpg")  # Wycinek obrazu 2
#
# # Wyodrębnij deskryptory
# descriptor1 = extract_descriptor(image1)
# descriptor2 = extract_descriptor(image2)
#
# # Porównaj deskryptory
# is_same_person, similarity_score = compare_descriptors(descriptor1, descriptor2)
# print(f"Czy to ta sama osoba? {'Tak' if is_same_person else 'Nie'}")
# print(f"Odległość kosinusowa: {similarity_score:.4f}")
