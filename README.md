# OVIAN
Ovarian Cancer AI Solution

## Introduction
OVIAN is an AI-powered solution for ovarian cancer pathology image analysis.  
It classifies cancer subtypes from histopathology images and provides PubMed-based literature retrieval with AI-generated summaries.  
(*This repository is prepared for competition submission. Trained model weights are not included.*)

## Environment Setup
### Using Conda
'''bash
conda env create -f ovian.yml
conda activate ovain

## Using pip
pip install -r requirements.txt

## How to run
streamlit run main.py
python flask_server.py

## Run Inference Script
### Perform subtype classification on a single image
### Can download sample image in Kaggle → UBC-OCEAN
python infer.py --input sample_image.png 

## Model Weights
### Pretrained model weights are not included in this repository.
### Please download from 'https://drive.google.com/file/d/1RdJm8dmDV3xhjPfKUXTmwaWoJLHN_t7B/view?usp=drive_link' and place them in the save_weights/ folder.

## Folder Structure
OVIAN/
├── .streamlit/        # Streamlit settings
├── Model training/    # Training scripts
├── models/            # Model definitions
├── save_weights/      # Trained weights (external)
├── main.py            # Streamlit app
├── flask_server.py    # Flask API server
├── infer.py           # Inference script
├── ovian.yml          # Conda environment file
├── requirements.txt   # Pip environment file
└── README.md
