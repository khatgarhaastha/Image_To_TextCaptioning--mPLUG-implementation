import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

from tqdm import tqdm

def load_model_and_tokenizer(model_name):
    # model is the encoder-decoder model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    # processor is used to process the images
    processor = ViTImageProcessor.from_pretrained(model_name)
    # tokenizer is used to process the captions
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, processor, tokenizer

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def predict_captions(model, processor, tokenizer, device, image_paths, max_length=16, num_beams=4):
    model.to(device)
    model.eval()
    images = [Image.open(img).convert("RGB") for img in image_paths]
    # get the pixel values for the images
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device) 
    # generate the captions
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    # decode the output ids to text
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # remove leading and trailing whitespaces
    return [caption.strip() for caption in captions]

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    with tqdm(total=len(data_loader)) as pbar:
        for idx, batch in (enumerate(data_loader)):
            inputs, targets = batch['image'].to(device), batch['input_ids'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs, labels=targets).logits.squeeze(0)
            targets = targets.squeeze(0)

            outputs = torch.permute(outputs, (0, 2, 1))
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'train_loss': total_loss / (idx+1)})
            pbar.update(1)

    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device, isVal= True):
    model.eval()
    total_loss = 0
    postFix = 'val_loss' if isVal else 'test_loss'
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for batch in data_loader:
                inputs, targets = batch['image'].to(device), batch['input_ids'].to(device)
                outputs = model(pixel_values=inputs, labels=targets).logits.squeeze(0)
                targets = targets.squeeze(0)
                outputs = torch.permute(outputs, (0, 2, 1))
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                pbar.set_postfix({postFix: total_loss / len(data_loader)})
                pbar.update(1)
    return total_loss / len(data_loader)
