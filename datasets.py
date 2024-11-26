import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IrisDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_mapping=None):
        """
        初始化数据集。
        - root_dir: 数据集的根目录。
        - transform: 图像预处理转换。
        - label_mapping: 类别名称到标签的映射字典。
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        self.label_mapping = label_mapping if label_mapping else {}
        self.image_paths = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        """
        加载数据集中的所有图像路径及其对应的标签。
        """
        persons = sorted(os.listdir(self.root_dir))

        # Loop through each person folder and assign labels
        for person in persons:
            person_dir = os.path.join(self.root_dir, person)
            if not os.path.isdir(person_dir):
                continue

            # Get the base name (first part before the underscore) to group similar people together
            base_name = '_'.join(person.split('_')[:2])  # 合并 person_1 和 person_1_1 为 person_1
            if base_name not in self.label_mapping:
                self.label_mapping[base_name] = len(self.label_mapping)  # 给每个新的人分配一个标签
            label = self.label_mapping[base_name]

            for eye in ['left_eye', 'right_eye']:
                eye_dir = os.path.join(person_dir, eye)
                if not os.path.exists(eye_dir):
                    continue
                for img_name in os.listdir(eye_dir):
                    img_path = os.path.join(eye_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Label mapping: {self.label_mapping}")
        print(f"Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 转为灰度图像
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
