# Import model from Scripts.model
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.data = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path and load image
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 1] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize the caption and obtain attention mask
        caption = self.data.iloc[idx, 2]
        tokenized_caption = self.tokenizer.encode_plus(
            f'{self.tokenizer.cls_token_id} {caption} {self.tokenizer.eos_token_id}',
            max_length=25,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        caption_ids = tokenized_caption['input_ids'].squeeze(0)  # Tensor of token ids
        attention_mask = tokenized_caption['attention_mask'].squeeze(0)  # Tensor of attention masks

        return image, caption_ids, attention_mask



def test():

    # Example usage
    img_dir = 'Data/instagram_data/'
    csv_file = 'Data/instagram_data/captions_csv.csv'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CaptionDataset(csv_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def create_dataloaders(csv_path, img_dir, transform, batch_size, tokenizer, test_size=0.2, val_size=0.1 ):
    data = pd.read_csv(csv_path)
    data = data[data.iloc[:, 2].notnull()]

    # for testing 
    data = data[:2000]
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)

    train_dataset = CaptionDataset(train_data, img_dir, tokenizer, transform)
    val_dataset = CaptionDataset(val_data, img_dir, tokenizer, transform)
    test_dataset = CaptionDataset(test_data, img_dir, tokenizer, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader