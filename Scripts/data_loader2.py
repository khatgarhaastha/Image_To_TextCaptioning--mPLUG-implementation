import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, AutoTokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, tokenizer, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            processor: Image processor (ViTImageProcessor).
            tokenizer: Tokenizer for processing captions (AutoTokenizer).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.caption_frame = pd.read_csv(csv_file)
        self.caption_frame = self.caption_frame.dropna()
        self.img_dir = img_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.caption_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.caption_frame.iloc[idx, 1] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Apply default processing using ViTImageProcessor
            image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        caption = self.caption_frame.iloc[idx, 2]
        # Tokenize captions here:
        tokenized_caption = self.tokenizer.encode(
            caption,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'image': image,
            'input_ids': tokenized_caption,
            #'attention_mask': tokenized_caption['attention_mask'].squeeze(0)
        }

def create_dataloaders(csv_path, img_dir, processor, tokenizer, batch_size=1, transform=None):
    dataset = ImageCaptionDataset(csv_file=csv_path, img_dir=img_dir, processor=processor, tokenizer=tokenizer, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
