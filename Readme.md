
# 🖼️ Image to Text Captioning using mPLUG

This project implements an **image captioning pipeline** that generates natural language descriptions from images using the **mPLUG** model. It leverages a pretrained vision-language transformer to understand visual content and produce contextually accurate captions.

---

## What is mPLUG?

**mPLUG** (Multimodal Pretraining with Language Understanding and Generation) is a transformer-based architecture that learns from both image and text data. It uses **cross-modal attention mechanisms** to align visual and textual embeddings, enabling it to perform tasks like image captioning, visual question answering, and image-text retrieval with high accuracy.

---

## Project Structure

```

.
├── data/
│   └── images/                 # Input images for captioning
├── src/
│   ├── model\_loader.py         # Loads the mPLUG model
│   ├── image\_preprocessor.py   # Handles image transformations
│   └── caption\_generator.py    # Generates captions from processed inputs
├── outputs/
│   └── generated\_captions.txt  # Stores output captions
├── app.py                      # Runs the full captioning pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation

````


## Model Architecture 

- Text Encoder 
  - We are using a text encoder model to generate vector representations for the caption data 
- Image Encoder 
  - We are using an Image Encoder model to generate the vector representation for the Images 
- Caption Decoder
  - This model would take inputs of Text encoder and Image Encoder and try to autoregressively generate the captions based on a cross-model attention-based architecture


![Architecture Diagram](path/to/your/image.png)


## Dataset 

- Kaggle Instagram Dataset @ https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions 

  - Dataset size: 4 GB 

  - Dataset Examples: 35,000
