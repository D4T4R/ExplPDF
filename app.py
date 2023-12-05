import streamlit as st 
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import base64           # to read the pdf
from PIL import Image

# Initialize the model & tokenizer
model_checkpoint = "LaMini-Flan-T5-248M"
custom_tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
summarization_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function for file loading and pre-processing
def process_file(input_file):
    document_loader = PyPDFLoader(input_file)
    document_pages = document_loader.load_and_split()
    text_chunker = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_chunker.split_documents(document_pages)
    combined_text = ""
    for chunk in chunks:
        combined_text += chunk.page_content
    return combined_text, len(combined_text)

# Summarization pipeline using a language model
def summarization_workflow(file_path):
    text_content, text_length = process_file(file_path)
    summarizer = pipeline(
        'summarization',
        model = summarization_model,
        tokenizer = custom_tokenizer,
        max_length = text_length//8, 
        min_length = 25)
    summarization_result = summarizer(text_content)
    summary_text = summarization_result[0]['summary_text']
    return summary_text

@st.cache_data      # Cache data to improve performance
def renderPDF(pdf_file):
    with open(pdf_file, "rb") as file:
        encoded_pdf = base64.b64encode(file.read()).decode('utf-8')
    embedded_pdf = F'<iframe src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(embedded_pdf, unsafe_allow_html=True)

# Streamlit user interface setup
st.set_page_config(page_title='Document Analyzer', layout="wide", page_icon="📄", initial_sidebar_state="expanded")
def app_main():
    st.title("Document Analyzer")
    logo_image = Image.open('document_logo.jpg')
    st.image(logo_image, width=200)

    file_upload = st.file_uploader("Upload a PDF Document", type=['pdf'])

    if file_upload is not None:
        if st.button("Generate Summary"):
            column_1, column_2 = st.columns([0.4,0.6])
            temp_filepath = "uploaded_docs/"+file_upload.name

            with open(temp_filepath, "wb") as file_buffer:
                file_buffer.write(file_upload.read())
            
            with column_1:
                st.info("PDF Preview")
                pdf_preview = renderPDF(temp_filepath)

            with column_2:
                document_summary = summarization_workflow(temp_filepath)
                st.info("Document Summary")
                st.success(document_summary)

# Initialize and run the Streamlit app
if __name__ == "__main__":
    app_main()
