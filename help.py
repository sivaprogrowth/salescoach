import streamlit as st
import requests
from util import *
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

@st.cache_data
def convert_pdf_to_txt_file(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return t 
# Step 1: Drop-down Selector
NEW_INDEX = "(New index)"
# Layout for the form 
options = get_index_list().names() + [NEW_INDEX]
option = st.selectbox("Select an option", options)

# Just to show the selected option
if option == NEW_INDEX:
    otherOption = st.text_input("Enter your other option...")
    if otherOption:
        selection = otherOption
        index_init(otherOption, 1536)
        st.info(f":white_check_mark: New index {otherOption} created! ")
    # Step 2: File Upload Block
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        if st.button("Send File"):
            name = uploaded_file.name
            raw_text = convert_pdf_to_txt_file(uploaded_file)
            response = requests.post("http://localhost:8000/createQuestionAnswer",json={"index": option, "text":raw_text})
            st.write(f"File upload status: {response.status_code}")
    option = otherOption
st.write(f"Selected option: {option}")