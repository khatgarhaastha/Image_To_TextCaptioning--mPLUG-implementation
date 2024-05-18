from tqdm import tqdm
from model import Model
from data_loader import create_dataloaders
import torch 
from torch.optim import Adam
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BertTokenizer

import wandb 

def train(model, data_loader, val_loader, device, criterion, optimizer, print_every=100):
    
    model.train()
    train_loss = 0.0

    for idx, (images, captions_ids, attention_ids) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        caption_ids = captions_ids.to(device)
        attention_ids = attention_ids.to(device)

        output = model(images, caption_ids, attention_ids)

        loss = criterion(output.view(-1, output.size(-1)), caption_ids.view(-1))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % print_every == 0:
            print(f'Batch: {idx}, Loss: {loss.item()}')

    model.eval()
    val_loss = 0.0

    for idx, (images, caption_ids, attention_ids) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        caption_ids = caption_ids.to(device)
        attention_ids = attention_ids.to(device)
    
        output = model(images, caption_ids, attention_ids)

        loss = criterion(output.view(-1, output.size(-1)), caption_ids.view(-1))
        val_loss += loss.item()
    
    return train_loss / len(data_loader), val_loss / len(val_loader)

def test(model, data_loader, device, criterion):
    
    model.eval()
    test_loss = 0.0

    for idx, (images, caption_ids, attention_ids) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        caption_ids = caption_ids.to(device)
        attention_ids = attention_ids.to(device)

        output = model(images, caption_ids, attention_ids)

        loss = criterion(output.view(-1, output.size(-1)), caption_ids.view(-1))
        '''
        Captions : [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
        Outputs : [seq_len, vocab_size]
        '''
        test_loss += loss.item()

    return test_loss / len(data_loader)


def generate_caption(model, images, tokenizer, device, max_length=50):
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        caption_ids = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)

        for _ in range(max_length):
            attention_ids = torch.ones((1, caption_ids.size(-1))).to(device)
            output = model(images, caption_ids, attention_ids)
            next_token = torch.argmax(output[0, -1, :])
            caption_ids = torch.cat([caption_ids, next_token.unsqueeze(0).view(1, 1)], dim=1)

            if next_token == tokenizer.encode('[SEP]')[0]:
                break

    return caption_ids

def main():
    model = Model(vocab_size=30522, image_size=2048, hidden_size=768, num_layers=1, nhead=2, dim_feedforward=128, dropout=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.00001)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, val_loader, test_loader = create_dataloaders('Data/instagram_data/captions_csv.csv', 'Data/instagram_data', transform=transform, batch_size=1, tokenizer=tokenizer)

    # for epoch in range(1):
    #     train_loss, val_loss = train(model, train_loader, val_loader, device, criterion, optimizer)
    #     print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # test_loss = test(model, test_loader, device, criterion)
    # print(f'Test Loss: {test_loss}')

    # Generate Captions 
    for idx, (images, caption_ids, attention_ids) in enumerate(test_loader):
        
        generated_caption_ids = generate_caption(model, images, tokenizer, device, max_length=50)
        generated_caption = tokenizer.decode(generated_caption_ids.squeeze(0).tolist())
        
        #Actual Caption
        caption = tokenizer.decode(caption_ids.squeeze(0).tolist())
        
        print(f'Actual Caption: {caption}')
        print(f'Generated Caption: {generated_caption}')
        print('-----------------------------------')




if __name__ == '__main__':
    main()
