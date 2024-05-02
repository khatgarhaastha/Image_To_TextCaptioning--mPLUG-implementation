import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel

from transformers import CLIP
class ImageCaptioning(nn.Module):
    def __init__(self, captionEncoder, imageProcessor, imageEncoder, CaptionDecoder):
        super(ImageCaptioning, self).__init__()
        self.imageProcessor = imageProcessor
        self.imageEncoder = imageEncoder
        self.captionEncoder = captionEncoder
        self.CaptionDecoder = CaptionDecoder


    def forward(self, images, captions):
        # Processing Images into Encodings 
        images = self.imageProcessor(images)
        imageFeatures = self.imageEncoder(images)

        # Processing Captions into Encodings
        captionFeatures = self.captionEncoder(captions)

        # Decoding the Captions
        outputs = self.CaptionDecoder(imageFeatures, captionFeatures)

        return outputs

# Define Image Encoder -> Resnet 
def getImageEncoder_Processor(modelName = "apple/mobilevit-small" ):
    processor = AutoImageProcessor.from_pretrained(modelName)
    model = AutoModelForImageClassification.from_pretrained(modelName)

    return processor, model

def getCaptionEncoder(modelName = "bert-base-uncased"):
    model = AutoModel.from_pretrained(modelName)
    return model

def getCaptionDecoder(modelName = "bert-base-uncased"):
    model = AutoModel.from_pretrained(modelName)
    return model