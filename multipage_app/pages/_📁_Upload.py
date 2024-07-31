import streamlit as st
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
#from IPython.display import display
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
import os
import json

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

top_image = Image.open('designTwo.jpg')
st.image(top_image)
st.title("Upload Image. Get Caption")
st.write(css, unsafe_allow_html=True)
uploaded_image = st.file_uploader(label="Add your image", type=['png', 'jpg', 'webp'])

class ImageCaptionTool(BaseTool):
    name = "Image caption Generator"
    description = "Use this tool When describing a given image."\
                    "It will explain the image based on a caption it generated."
    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=250)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
tools = [ImageCaptionTool()]
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name="gpt-4o-mini"
)
agent = initialize_agent(
    agent = "chat-conversational-react-description",
    tools = tools,
    llm = llm,
    verbose = True,
    early_stopping_method = 'generate'
)
if uploaded_image is None:
    st.write("Please upload an image.")
else:
    st.image(uploaded_image)
    bytes_data = uploaded_image.read()

    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")

    with open(os.path.join("tempDir", uploaded_image.name), "wb") as f:
        f.write(bytes_data)
    
    file_path = os.path.join("tempDir", uploaded_image.name)

    context = st.text_area(label="Additional Context (Optional)", height=100)
    if st.button("Generate Image Caption"):
        with st.spinner("Generating..."):
            if context:
                output = agent.run(
                    input=(f"Considering this context: {context}\nGenerate a caption for this image: {file_path}")
                )
            else:
                output = agent.run({"input": f"Generate a caption for this image: {file_path}", "chat_history": []})
            st.markdown(bot_template.replace("{{MSG}}", output), unsafe_allow_html=True)