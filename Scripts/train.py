from tqdm import tqdm
from model import Model
from data_loader import create_dataloaders
import torch 
from torch.optim import Adam
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BertTokenizer

def train(model, data_loader, val_loader, device, criterion, optimizer, print_every=100):
    
    model.train()
    train_loss = 0.0

    for idx, (images, captions) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        captions = captions.to(device)

        output = model(images, captions)

        loss = criterion(output.view(-1, output.size(-1)), captions.view(-1))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % print_every == 0:
            print(f'Batch: {idx}, Loss: {loss.item()}')

    model.eval()
    val_loss = 0.0

    for idx, (images, captions) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        captions = captions.to(device)

        output = model(images, captions)

        loss = criterion(output.view(-1, output.size(-1)), captions.view(-1))
        val_loss += loss.item()
    
    return train_loss, val_loss

def test(model, data_loader, device, criterion):
    
    model.eval()
    test_loss = 0.0

    for idx, (images, captions) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        captions = captions.to(device)

        output = model(images, captions)

        loss = criterion(output.view(-1, output.size(-1)), captions.view(-1))
        test_loss += loss.item()

    return test_loss

def main():
    model = Model(vocab_size=1000, image_size=2048, hidden_size=512, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, val_loader, test_loader = create_dataloaders('Data/instagram_data/captions_csv.csv', 'Data/instagram_data/', transform=transform, batch_size=32, tokenizer=tokenizer)

    for epoch in range(10):
        train_loss, val_loss = train(model, train_loader, val_loader, device, criterion, optimizer)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    test_loss = test(model, test_loader, device, criterion)
    print(f'Test Loss: {test_loss}')


if __name__ == '__main__':
    main()
