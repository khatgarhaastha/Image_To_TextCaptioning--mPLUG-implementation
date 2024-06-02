from data_loader2 import create_dataloaders
from model2 import load_model_and_tokenizer, setup_device, train_one_epoch, validate
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

import wandb 

def main():
    # Path setup
    csv_file = 'Data/instagram_data/captions_csv.csv'
    img_dir = 'Data/instagram_data/'
    
    # Model and Tokenizer setup
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model, processor, tokenizer = load_model_and_tokenizer(model_name)
    device = setup_device()
    model.to(device)
    
    # Hyper parameters
    learning_rate = 0.001
    batch_size = 1
    epochs = 5

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, val_loader, test_loader = create_dataloaders(csv_file, img_dir, processor, tokenizer = tokenizer, batch_size=batch_size, transform=transform)
    
    wandb.init(project='instagram-captionning', config = {"architecture": "ViT-GPT2", "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})
    # Training and validation loop
    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        test_loss = validate(model, test_loader, criterion, device, isVal = False)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

        wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Test Loss": test_loss})
        # Save the model
        torch.save(model.state_dict(), f'Saved_models/VIT/model_{epoch}.pth')

if __name__ == '__main__':
    main()
