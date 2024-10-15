# Multimodel App

This is a Streamlit-based web application that provides various AI-driven generation functionalities, including:
- Text-to-Image
- Image-to-Text
- Text-to-Audio
- Image-to-Image

Each functionality utilizes different pre-trained models from Hugging Face and Diffusers, enabling users to easily generate and transform media.

## Features

- **Text-to-Image**: Generates images from text prompts using stable diffusion models.
- **Image-to-Text**: Captions images with descriptions using an image-captioning model.
- **Text-to-Audio**: Generates audio/music from text prompts using the MusicGen model.
- **Image-to-Image**: Transforms images based on different image manipulation techniques.

## Installation

### Prerequisites
- Python 3.10+
- Conda or virtualenv for creating a virtual environment

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/rishurnjan/multimodel-app.git
   cd multimodel-app
2. Create and activate new conda env
   - conda create -n mm_env python=3.10
     conda activate mm_env
3. Install required dependencies
   - pip install -r requirement.txt
4. Run the app
   - streamlit run app.py 

Usage
Start the app using the command above.
Use the sidebar menu to navigate between different options.
For each task, input the required data (text prompt, image upload, etc.) and click the respective button to generate output.  

Model Details
Text-to-Image: Uses the stable diffusion model (CompVis/stable-diffusion-v1-4) to generate images from text prompts.
Image-to-Text: Uses the nlpconnect/vit-gpt2-image-captioning model for generating captions from images.
Text-to-Audio: Uses the facebook/musicgen-small model to generate audio from text prompts.
Image-to-Image: Utilizes diffusion pipelines for image transformations.
            
