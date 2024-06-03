import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, AutoTokenizer
from sklearn.model_selection import train_test_split
class ImageCaptionDataset(Dataset):
    def __init__(self, caption_frame, img_dir, processor, tokenizer, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            processor: Image processor (ViTImageProcessor).
            tokenizer: Tokenizer for processing captions (AutoTokenizer).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.caption_frame = caption_frame
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
            f'{self.tokenizer.cls_token_id} {caption} {self.tokenizer.eos_token_id}',
            add_special_tokens=True,
            max_length=25,
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
    caption_frame = pd.read_csv(csv_path)
    caption_frame = caption_frame.dropna()

    # for testing purposes
    #caption_frame = caption_frame[:11]
    test_size = 0.2
    val_size = 0.1

    #split data into train and test
    train_data, test_data = train_test_split(caption_frame, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)


    train_dataset = ImageCaptionDataset(caption_frame=caption_frame, img_dir=img_dir, processor=processor, tokenizer=tokenizer, transform=transform)
    val_dataset = ImageCaptionDataset(caption_frame=val_data, img_dir=img_dir, processor=processor, tokenizer=tokenizer, transform=transform)
    test_dataset = ImageCaptionDataset(caption_frame=test_data, img_dir=img_dir, processor=processor, tokenizer=tokenizer, transform=transform)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
