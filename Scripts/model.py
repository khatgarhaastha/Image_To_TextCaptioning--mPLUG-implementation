import torch 
from transformers import BertModel
from transformers import AutoImageProcessor, AutoModel

class Model(torch.nn.Module):
    def __init__(self, vocab_size, image_size, hidden_size, num_layers, nhead, dim_feedforward, dropout):
        super(Model, self).__init__()

        #self.image_encoder = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")
        self.image_linear = torch.nn.Linear(768, hidden_size)
        self.text_linear = torch.nn.Linear(768, hidden_size)

        #self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder = torch.nn.Embedding(vocab_size, 768)
        self.transformer = torch.nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, images, caption_ids, attention_ids):
        #image_features = self.Image_encoder(images).last_hidden_state
        image_features = self.image_processor(images, return_tensors="pt").pixel_values.to(images.device)
        image_features = self.image_encoder(image_features).pooler_output
        image_features = image_features.unsqueeze(0)
        


        caption_embeddings = self.text_encoder(caption_ids)
        caption_embeddings = caption_embeddings.permute(1, 0, 2)
        tgt_mask = self.transformer.generate_square_subsequent_mask(caption_ids.size(-1)).to(caption_ids.device)
        transformer_output = self.transformer(src=image_features, tgt=caption_embeddings, tgt_mask=tgt_mask)
        output = self.output_projection(transformer_output)

        return output
    

    
    
        