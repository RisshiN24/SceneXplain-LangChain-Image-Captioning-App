import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Caption Generator",
    page_icon="🖊️"
)

st.title("Image Caption Generator")
st.write("Powered by GenAI")
st.sidebar.success("Select a page above.")

top_image = Image.open('homeImage.jpg')
st.image(top_image)