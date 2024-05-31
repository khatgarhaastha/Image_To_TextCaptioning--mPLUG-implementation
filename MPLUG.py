import torch 
import torch.nn as nn

class cross_layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(cross_layer,self).__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm_1 = nn.LayerNorm(embed_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm_2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Linear(embed_dim, embed_dim)

    def forward(self, image_features, text_features):
        text_features_processed ,_= self.self_attention(text_features, text_features, text_features)
        text_features = self.norm_1(text_features + text_features_processed)
        cross_features_processed ,_= self.cross_attention(text_features, image_features, image_features)
        text_features = self.norm_2(text_features + cross_features_processed)
        ffn_features = self.ffn(text_features)
        return image_features, ffn_features

class connected_layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(connected_layer,self).__init__()
        self.embed_dim = embed_dim 
        self.self_attention = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=0.1)
        self.norm_1 = nn.LayerNorm(self.embed_dim)

        self.ffn = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm_2 = nn.LayerNorm(self.embed_dim)

    def forward(self, image_features, text_features):
        concated_features = torch.cat((image_features, text_features), dim=0)
        concated_features2,_ = self.self_attention(concated_features, concated_features, concated_features)
        concated_features = self.norm_1(concated_features + concated_features2)

        ffn_features2 = self.ffn(concated_features)
        concated_features = self.norm_2(concated_features + ffn_features2)

        # split the concated features into image and text features
        image_features = concated_features[0, :].unsqueeze(0)
        text_features = concated_features[1:, :]
        return image_features, text_features

class encoder_layer(nn.Module):
    def __init__(self, embed_dim, num_heads, skip_layer_numbers):
        super(encoder_layer, self).__init__()
        self.image_linaar = nn.Linear(1000, embed_dim)
        self.cross_layers = nn.ModuleList([cross_layer(embed_dim, num_heads) for _ in range(skip_layer_numbers)])
        self.connected_layer = connected_layer(embed_dim, num_heads)

    def forward(self, image_features, text_features):
        for cross_layer in self.cross_layers:
            image_features, text_features = cross_layer(image_features, text_features)

        image_features, text_features = self.connected_layer(image_features, text_features)

        return image_features, text_features
    
class decoder_layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(decoder_layer, self).__init__()
        
        # define a causal self attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm_1 = nn.LayerNorm(embed_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm_2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Linear(embed_dim, embed_dim)


    def forward(self, concatenated_features, text_features):
        seq_length = text_features.size(0)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(text_features.device)
        text_features_processed,_ = self.self_attention(text_features, text_features, text_features, attn_mask=causal_mask)
        text_features = self.norm_1(text_features + text_features_processed)
        cross_features_processed , _ = self.cross_attention(text_features, concatenated_features, concatenated_features)
        text_features = self.norm_2(text_features + cross_features_processed)

        ffn_features = self.ffn(text_features)
        return concatenated_features, ffn_features
        
class MPLUG_Implementation(nn.Module):
    def __init__(self, embed_dim, num_heads, skip_layer_numbers, encoder_layers_number, decoder_layers_number):
        super(MPLUG_Implementation, self).__init__()

        # define Encoder Layers
        self.encoder_layers = nn.ModuleList([encoder_layer(embed_dim, num_heads, skip_layer_numbers) for _ in range(encoder_layers_number)])    

        # Define the decoder Layers 
        self.decoder_layers = nn.ModuleList([decoder_layer(embed_dim, num_heads) for _ in range(decoder_layers_number)])

        self.image_linaar = nn.Linear(1000, embed_dim)

        self.final_linear = nn.Linear(embed_dim, 30522)
    def forward(self, image_features, text_features):
        # Implement the forward pass
        image_features = self.image_linaar(image_features)
        for encoder_layer in self.encoder_layers:
            image_features, text_features = encoder_layer(image_features, text_features)

        concatenated_features = torch.cat((image_features, text_features), dim=0)
        #create causal mask for text features
        
        for decoder_layer in self.decoder_layers:
            _, decoder_output = decoder_layer(concatenated_features, text_features)

        decoder_output = self.final_linear(decoder_output)
        return decoder_output

        
