import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import joblib
import cv2

model = joblib.load("model.pkl")

st.title("Rita en siffra")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw"
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data[:, :, 0]

    img = Image.fromarray(img.astype('uint8')).convert("L")

    img = np.array(img)

    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(img > 0))
    if coords.size > 0:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        img = img[x0:x1, y0:y1]

    img = cv2.resize(img, (28, 28))

    img = img / 255.0

    st.image(img, caption="Vad modellen ser", width=150)

    img = img.reshape(1, -1)

    prediction = model.predict(img)
    proba = model.predict_proba(img)

    st.write("Prediktion:", prediction[0])
    st.bar_chart(proba[0])

    