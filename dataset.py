from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

classes = ["Airport", "Bridge", "Center", "Desert", "Forest",
    "Industrial", "Mountain", "Pond", "Port", "Stadium"]

class AIDataset(Dataset):

    def __init__(self, csv_file_path, transform = None):
        self.data_df = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]["paths"]
        target_id = self.data_df.iloc[idx]["labels"]

        img = Image.open(img_path)

        if self.transform:
            img_tensor = self.transform(img)

        return (img_tensor, target_id)

    def _save_csv(self, path):
        self.data_df.to_csv(path, sep=",")



