import torch 
from transformers import VisualBertModel, BertModel
from transformers import AutoImageProcessor, ResNetForImageClassification

class Model(torch.nn.Module):
    def __init__(self, vocab_size, image_size, hidden_size, num_layers, nhead, dim_feedforward, dropout):
        super(Model, self).__init__()

        #self.image_encoder = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.image_encoder = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.image_linear = torch.nn.Linear(image_size, hidden_size)
        self.text_linear = torch.nn.Linear(768, hidden_size)

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.transformer = torch.nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        print(images.shape)
        print(captions.shape)
        
        #image_features = self.Image_encoder(images).last_hidden_state
        image_features = self.image_processor(images, return_tensors="pt", do_rescaling=False)
        image_features = self.image_encoder(image_features.pixel_values).last_hidden_state
        caption_embeddings = self.text_encoder(captions)[0]
        image_features = self.image_linear(image_features)
        caption_embeddings = self.text_linear(caption_embeddings)

        tgt_mask = self.transformer.generate_square_subsequent_mask(captions.size(-1)).to(captions.device)
        transformer_output = self.transformer(src=image_features, tgt=caption_embeddings, tgt_mask=tgt_mask)
        output = self.output_projection(transformer_output)

        return output
    
    
        