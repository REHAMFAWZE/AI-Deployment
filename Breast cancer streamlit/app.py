import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# ---- Load model ----
model = load_model("segmentation_model.keras", compile=False)

# ---- Function to preprocess image ----
def prepare_image(uploaded_file):
    # افتح الصورة من Streamlit uploader
    img = Image.open(uploaded_file).convert("L")   # غرايسكيل
    img = img.resize((256, 256))                   # Resize
    
    # Convert to numpy + normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # أضف channel و batch
    img_array = np.expand_dims(img_array, axis=-1)   # (256,256) -> (256,256,1)
    img_array = np.expand_dims(img_array, axis=0)    # -> (1,256,256,1)
    
    return img_array, img   # نرجع النسخة numpy + نسخة PIL

# ---- Streamlit UI ----
st.title("Medical Image Segmentation App 🩺")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # prepare image
    img_array, processed_img = prepare_image(uploaded_file)

    # predict
    pred_mask = model.predict(img_array)[0]

    # post-process mask
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask.squeeze())

    st.image(processed_img, caption="Processed Input (256x256 Grayscale)")
    st.image(mask_img, caption="Predicted Mask", use_column_width=True)

    # download button for processed image
    processed_img.save("processed_input.png")
    with open("processed_input.png", "rb") as file:
        st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="processed_input.png",
            mime="image/png"
        )
