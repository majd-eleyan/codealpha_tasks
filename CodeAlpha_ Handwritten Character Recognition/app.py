import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(page_title="Digit Recognition", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit with your mouse or upload an image, and the model will try to predict it.")

# Load the model once to avoid slowing down the app
@st.cache_resource
def load_my_model():
    # 
    # ✅✅✅✅✅✅✅ اعمل المسار للملف
    # 
    model = tf.keras.models.load_model("C:\\Users\\Tamer\\Desktop\\2\\handwritten_digit_model.h5")
    return model

# Attempt to load the model, stop with an error message if it fails
try:
    model = load_my_model()
except Exception as error:
    st.error("Could not load the model. Ensure 'handwritten_digit_model.h5' is in the same directory.")
    st.stop()

def prepare_image(img):
    """
    Prepares any image (from canvas or upload)
    to match model requirements: 28x28 size and 0-1 range.
    """
    # Convert RGBA/RGB to Grayscale
    if img.shape[-1] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Invert colors if the background is light (MNIST is white on black)
    if gray.mean() > 127:
        gray = 255 - gray

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize values between 0 and 1
    normalized = resized / 255.0

    # Reshape for the model (1, 28, 28, 1)
    final_image = normalized.reshape(1, 28, 28, 1)

    return final_image, resized

def show_prediction(final_image, thumbnail):
    """Displays prediction results with confidence level and chart"""
    if final_image.max() == 0:
        st.info("Please draw a digit or upload an image first")
        return

    prediction = model.predict(final_image)
    predicted_number = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Digit", predicted_number)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.image(thumbnail, caption="Processed Image (28x28)", width=150)

    with col2:
        st.write("Probability per Digit:")
        st.bar_chart(prediction[0])

    if confidence < 60:
        st.warning("Low confidence. Try drawing the digit more clearly in the center.")

# Tabs for Drawing and Uploading
tab1, tab2 = st.tabs(["🖌️ Draw Digit", "📤 Upload Image"])

with tab1:
    st.write("Draw a single digit in the box below")
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        drawn_image = canvas_result.image_data.astype(np.uint8)
        final_image, thumbnail = prepare_image(drawn_image)
        show_prediction(final_image, thumbnail)

with tab2:
    st.write("Upload an image containing a clear single digit")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            np_image = np.array(pil_image)
            st.image(pil_image, caption="Uploaded Image", width=200)
            final_image, thumbnail = prepare_image(np_image)
            show_prediction(final_image, thumbnail)
        except Exception as error:
            st.error("Error processing the uploaded image. Please try another.")