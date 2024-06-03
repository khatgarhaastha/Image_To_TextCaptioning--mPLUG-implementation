import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoImageProcessor, ResNetForImageClassification, BertModel

class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, text_tokenizer, image_processor, image_encoder, text_encoder, transform=None):
        self.data = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_embedding = None
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.image_embedding = output
            print("Hook called. Captured output:", output.shape)
        
        # Register hook to the last hidden layer before the classifier
        for name, module in self.image_encoder.named_modules():
            if name == 'resnet.layer4.2':  # Adjust this to the correct layer
                module.register_forward_hook(hook)
                print(f"Hook registered to layer: {name}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path and load image
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 1] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
    
        # Preprocess the image
        inputs = self.image_processor(image, return_tensors="pt")
        
        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = self.image_encoder(inputs.pixel_values).logits
        # Tokenize the caption and obtain attention mask
        caption = self.data.iloc[idx, 2]
        tokenized_caption = self.tokenizer(caption, 
                                           return_tensors='pt',
                                           max_length=25,
                                           truncation=True, 
                                           padding='max_length')
        caption_ids = tokenized_caption['input_ids']
        attention_mask = tokenized_caption['attention_mask']

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=caption_ids, attention_mask=attention_mask)
            caption_embeddings = outputs.last_hidden_state.squeeze(0)
        
        # Convert all tensors to float32
        caption_embeddings = caption_embeddings.float()
        attention_mask = attention_mask.squeeze(0)
        return image_embeddings, caption_embeddings, attention_mask, tokenized_caption['input_ids']

