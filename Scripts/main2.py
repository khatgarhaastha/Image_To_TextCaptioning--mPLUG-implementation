from data_loader2 import create_dataloaders
from model2 import load_model_and_tokenizer, setup_device, train_one_epoch, validate
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

def main():
    # Path setup
    csv_file = 'Data/instagram_data/captions_csv.csv'
    img_dir = 'Data/instagram_data/'
    
    # Model and Tokenizer setup
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model, processor, tokenizer = load_model_and_tokenizer(model_name)
    device = setup_device()
    model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = create_dataloaders(csv_file, img_dir, processor, tokenizer = tokenizer, batch_size=1, transform=transform)
    val_loader = create_dataloaders(csv_file, img_dir, processor,tokenizer = tokenizer, batch_size=1, transform=transform)  # Adjust path as needed

    # Training and validation loop
    num_epochs = 1
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()
