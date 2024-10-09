---
title: Hello Sexy!

---

Hello Sexy!






Problem statement:
Implemented a simple implementation of mPLUG <LINK HERE> for research Purpose. 



Methodology : 
2 base models 

-> Basic transformer that takes Image Embeddings (RESNET Classifier) and Caption Embeddings (BERT-base-uncased) as input and output embeddings 


-> ViT : Vision Transformer -> Pretrained

:: READ MORE ABOUT VIT ARCHITECTURE 

Actual Model : mPLUG 

-> Uses Cross Modal Attention Mechanism to extract added information from text inputs. 

Basic Idea is : 

In general Image embeddings are very feature rich and TExt not so much -> On account of them being smaller in size. 
Not much is extracted from the Text embeddings. 

So we do additional Training steps on Text embeddings while crossing the embeddings with Image Embeddings in order to extract greater amount of information from Text 


So in the Cross Attention Module, we have : 

- Inputs of Image and Caption Embeddings 

- for N times, we do cross attention of Text and Image Embeddings to enrich the quality of Text Embeddings while the Image is enriched in the final layer of the Encoder Block.

- Use this enriched Image encoding for the decoder layer. 



Results (lol)

- 18 for Transformer Base 
- 21 for Vision Transformer Pre-trained
- 23 for mPLUG implementation -> 27% increase in performance 

Here’s a concise version for your resume:

- Developed and implemented mPLUG model integrating Cross-Modal Attention to enhance text embeddings using image embeddings.
- Compared performance across Transformer (ResNet + BERT) and Vision Transformer (ViT) base models, achieving a 27% performance improvement with mPLUG.
- Utilized PyTorch, BERT-base, ResNet, and Vision Transformer (ViT) for cross-modal understanding and attention mechanism design. 

This format keeps it short and impactful!