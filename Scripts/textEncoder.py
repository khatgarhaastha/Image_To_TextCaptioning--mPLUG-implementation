"""
This part of the code is to create the encoder for the cpations.
The captions are tokenized using the BERT tokenizer.
the images and captions are then passed through the model to get the hidden states.
The hidden states are then used to generate the caption.

"""


import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

'''
Some of the captions are exceeding the token limit of BERT.
We are going to split the captions into multiple parts and encode them separately.
This will allow us to encode the entire caption without losing any information.
'''


# Function to split and encode text
def split_encode(text):
    token_ids = tokenizer.encode(text, add_special_tokens=True)  # Tokenize the text
    max_length = 512  # BERT's maximum sequence length
    chunks = [token_ids[i:i + max_length] for i in range(0, len(token_ids), max_length)]
    """
    for chunk in chunks:
        print(f"Chunk length: {len(chunk)}")  # Print each chunk length
    assert all(len(chunk) <= 512 for chunk in chunks), "One or more chunks exceed the maximum length"
    """
    # Add padding to the chunks that are less than the maximum length
    # chunks = [chunk + [0] * (max_length - len(chunk)) for chunk in chunks]
    return chunks


# Load your data
data_file = 'Data/instagram_data/captions_csv.csv'
data = pd.read_csv(data_file)


# Handling missing values by removing or filling
data['Caption'] = data['Caption'].fillna('')  # Fill NaN with empty strings

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the captions using the split_encode function
data['tokens'] = data['Caption'].apply(split_encode)

# Print the tokenized data to check
#print(data['tokens'].head())

# Creating the dataset 
class captions_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return the tokenized captions at the index
        return torch.tensor(self.captions[idx], dtype=torch.long)
    
# Initialize the dataset
dataset = captions_dataset(data['tokens'])

def collate_batch(batch):
    # Padding the batch to the longest sequence in the batch
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch_padded

# Create the DataLoader for batching data
caption_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)



