import streamlit as st
from PIL import Image
from io import BytesIO
from image_inference import detect_and_annotate
import os

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Human-Heron Detector")
st.write(
    ":dog: Try uploading an image to detect person or Heron. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Directory where pre-recorded images are stored
PRE_RECORDED_DIR = "pre_recorded_images"

# List available pre-recorded images
pre_recorded_images = [f for f in os.listdir(PRE_RECORDED_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(pre_recorded_images)


# Download the fixed image
def convert_image(img):
    annotated_image_rgb = Image.fromarray(img)
    buf = BytesIO()
    annotated_image_rgb.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


# Function to process the uploaded image and apply detection
def process_image(upload, key):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    # Detect and annotate the image
    annotated_image = detect_and_annotate(image)

    col2.write("Annotated Image :wrench:")
    col2.image(annotated_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download annotated image", convert_image(annotated_image), "annotated.png", "image/png", key=key)

# Streamlit layout for image upload and columns
col1, col2 = st.columns(2)

# ----------------------------------------------------------------
# Option to select a pre-recorded image
selected_image_name = st.sidebar.selectbox("Select a pre-recorded image", pre_recorded_images)

# Load selected pre-recorded image
if selected_image_name:
    selected_image_path = os.path.join(PRE_RECORDED_DIR, selected_image_name)
    process_image(selected_image_path, key="pre-recorded")

# ----------------------------------------------------------------
# Option to upload an image

my_upload = st.sidebar.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        process_image(upload=my_upload, key="uploaded")
else:
    st.write("Please upload an image to start detection.")

# ----------------------------------------------------------------

# Option to capture image from webcam
st.sidebar.write("## Or use your webcam")
webcam_image = st.camera_input("Take a picture")

if webcam_image:
    process_image(webcam_image, key="webcam")