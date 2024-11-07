import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class IrisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir: Root directory of the dataset.
        - transform: Image preprocessing transformations.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_mapping = {}
        self._load_dataset()

    def _load_dataset(self):
        """
        Load all image paths and their corresponding labels from the dataset.
        """
        persons = os.listdir(self.root_dir)
        for idx, person in enumerate(persons):
            person_dir = os.path.join(self.root_dir, person)
            if not os.path.isdir(person_dir):
                continue
            self.label_mapping[person] = idx
            for eye in ['left_eye', 'right_eye']:
                eye_dir = os.path.join(person_dir, eye)
                if not os.path.exists(eye_dir):
                    continue
                for img_name in os.listdir(eye_dir):
                    img_path = os.path.join(eye_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the sample with the specified index.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
