import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline
import soundfile as sf
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image

# Function to handle text-to-image generation
@st.cache_resource
def load_text_to_image_model():
    return DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

def text_to_image():
    st.header("Text to Image")
    prompt = st.text_input("Enter text to generate an image:")
    model = load_text_to_image_model()
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            result = model(prompt).images[0]
            st.image(result, caption="Generated Image", use_column_width=True)

# Function to handle image-to-text generation
@st.cache_resource
def load_image_to_text_model():
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def image_to_text():
    st.header("Image to Text")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    model = load_image_to_text_model()
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text..."):
            description = model(image)[0]["generated_text"]
            st.write("Extracted Text:", description)
    else:
        st.warning("Please upload an image.")

# Function to handle text-to-audio generation
@st.cache_resource
def load_text_to_audio_model():
    return pipeline("text-to-audio", model="facebook/musicgen-small")

def text_to_audio():
    st.header("Text to Audio")
    text_input = st.text_input("Enter text to convert to audio:")
    model = load_text_to_audio_model()
    if st.button("Generate Audio"):
        with st.spinner("Generating audio..."):
            audio_data = model(text_input)["audio"][0]
            audio_path = "generated_audio.wav"
            sf.write(audio_path, np.array(audio_data), 16000)
            st.audio(audio_path, format="audio/wav")

# Function to handle image-to-image generation
@st.cache_resource
def load_image_to_image_model():
    return DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

def image_to_image():
    st.header("Image to Image")
    uploaded_file = st.file_uploader("Upload an image for transformation", type=["jpg", "jpeg", "png"])
    model = load_image_to_image_model()
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        with st.spinner("Transforming image..."):
            transformed_image = model(prompt="style transformation prompt", init_image=image).images[0]
            st.image(transformed_image, caption="Transformed Image", use_column_width=True)

# Main Streamlit app layout
st.set_page_config(page_title="AI Generation Web App", layout="centered")
st.title("AI Generation Web App")

# Sidebar menu for selecting task
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu", 
        options=["Home", "Text to Image", "Image to Text", "Text to Audio", "Image to Image"],
        icons=["house", "image", "file-text", "music-note", "shuffle"],
        menu_icon="cast", 
        default_index=0,
    )

# Navigation based on user selection
if selected == "Home":
    st.write("### Select a task from the sidebar to get started.")
elif selected == "Text to Image":
    text_to_image()
elif selected == "Image to Text":
    image_to_text()
elif selected == "Text to Audio":
    text_to_audio()
elif selected == "Image to Image":
    image_to_image()
