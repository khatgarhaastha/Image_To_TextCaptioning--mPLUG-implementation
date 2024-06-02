
from sklearn.model_selection import train_test_split
from Mplug_dataloader import CaptionDataset
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.optim import Adam
from MPLUG import MPLUG_Implementation
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification , ResNetConfig
import wandb 

def train(model, data_loader, val_loader, device, criterion, optimizer, print_every=100):

    model.train()
    train_loss = 0.0

    with tqdm(total=len(data_loader)) as pbar:
        for idx, (images, captions_ids, attention_ids, target_ids) in (enumerate(data_loader)):
            images = images.to(device)
            caption_ids = captions_ids.to(device)
            attention_ids = attention_ids.to(device)
            target_ids = target_ids.to(device)

            images = images.permute(1,0,2)

            caption_ids = caption_ids.permute(1,0,2)

            output = model(images, caption_ids)

            loss = criterion(output.view(-1, 30522), target_ids.view(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % print_every == 0:
                print(f'Batch: {idx}, Loss: {loss.item()}')

            pbar.set_postfix({'train_loss': train_loss / len(data_loader)})
            pbar.update(1)

    model.eval()
    val_loss = 0.0

    with tqdm(total=len(val_loader)) as pbar:
        for idx, (images, caption_ids, attention_ids, target_ids) in (enumerate(val_loader)):
            images = images.to(device)
            caption_ids = caption_ids.to(device)
            attention_ids = attention_ids.to(device)
            target_ids = target_ids.to(device)
            images = images.permute(1,0,2)

            caption_ids = caption_ids.permute(1,0,2)
            
            output = model(images, caption_ids)

            loss = criterion(output.view(-1, 30522), target_ids.view(-1))
            val_loss += loss.item()

            pbar.set_postfix({'val_loss': val_loss / len(val_loader)})
            pbar.update(1)
        
    return train_loss / len(data_loader) , val_loss / len(val_loader)

def test(model, data_loader, device, criterion):

    model.eval()
    test_loss = 0.0

    with tqdm(total=len(data_loader)) as pbar:
        for idx, (images, caption_ids, attention_ids, target_ids) in tqdm(enumerate(data_loader)):
            images = images.to(device)
            caption_ids = caption_ids.to(device)
            attention_ids = attention_ids.to(device)
            target_ids = target_ids.to(device)
            # permute images 31,1,1000 -> 1,32,1000
            images = images.permute(1,0,2)

            caption_ids = caption_ids.permute(1,0,2)
            output = model(images, caption_ids)

            loss = criterion(output.view(-1, 30522), target_ids.view(-1))
            test_loss += loss.item()
            
            pbar.set_postfix({'test_loss': test_loss / len(data_loader)})
            pbar.update(1)
            
    
    return test_loss / len(data_loader)

def create_dataloaders(csv_path, img_dir, transform, batch_size, text_tokenizer,image_processor, image_encoder, text_encoder, test_size=0.2, val_size=0.1 ):
    data = pd.read_csv(csv_path)
    data = data[data.iloc[:, 2].notnull()]

    # for testing purposes
    data = data[:2000]

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)
    '''
    csv_file, img_dir, text_tokenizer, image_processor, image_encoder, transform=None
    '''
    train_dataset = CaptionDataset(train_data, img_dir, text_tokenizer, image_processor, image_encoder, text_encoder, transform)
    val_dataset = CaptionDataset(val_data, img_dir, text_tokenizer, image_processor, image_encoder, text_encoder, transform)
    test_dataset = CaptionDataset(test_data, img_dir, text_tokenizer, image_processor, image_encoder, text_encoder, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
def main():
    # Hyperparameters
    batch_size = 1
    learning_rate = 0.001
    epochs = 4
    embed_dim = 768
    num_heads = 4
    skip_layer_numbers = 3
    encoder_layers_number = 3
    decoder_layers_number = 2

    model = MPLUG_Implementation(embed_dim, num_heads, skip_layer_numbers, encoder_layers_number, decoder_layers_number)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = BertModel.from_pretrained('bert-base-uncased')

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    image_encoder_config = ResNetConfig.from_pretrained("microsoft/resnet-50", output_hidden_states=True)
    image_encoder = ResNetForImageClassification.from_pretrained("microsoft/resnet-50",config = image_encoder_config)

    print(image_encoder)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        img_dir='Data/instagram_data',
        csv_path='Data/instagram_data/captions_csv.csv',
        text_tokenizer=text_tokenizer,
        transform=transform,
        batch_size=batch_size,
        image_processor=image_processor,
        image_encoder=image_encoder,
        text_encoder = text_encoder
    )


    # init wandb 

    wandb.init(project='instagram-captionning', config = {"architecture": "MPLUG", "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs, "embed_dim": embed_dim, "num_heads": num_heads, "skip_layer_numbers": skip_layer_numbers, "encoder_layers_number": encoder_layers_number, "decoder_layers_number": decoder_layers_number})
    wandb.watch(model, log='all')
    
    for epoch in range(epochs):
        train_loss, val_loss = train(model, train_dataloader, val_dataloader, device, criterion, optimizer)
        test_loss = test(model, test_dataloader, device, criterion)

        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}')

        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Test Loss": test_loss})
        # Save Model
        torch.save(model.state_dict(), f'Saved_models/MPLUG/model_{epoch}.pth')

if __name__ == '__main__':
    main()
