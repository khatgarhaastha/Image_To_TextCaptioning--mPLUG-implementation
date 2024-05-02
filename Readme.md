# Make Captions Great Again
This is a Complete E2E app that will allow the users to upload their Instagram photos and be able to generate captions for their photos.


## Dataset 

- Kaggle Instagram Dataset @ https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions 

  - Dataset size : 4 GB 

  - Dataset Examples : 35,000

## Model Architecture 

- Text Encoder 
  - We are using a text encoder model to generate vector representations for the caption data 
- Image Encoder 
  - We are using a Image Encoder model to generate the vectore representation for the Images 
- Caption Decoder
  - This model would take inputs of Text encoder and Image Encoder and try to autoregressively generate the captions based on a cross attention based architecture


## Setup Instructions 

- Install Dependencies 

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install instaloader 