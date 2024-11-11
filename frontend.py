import streamlit as st
import requests
from util import *
import queue, wave
import sounddevice as sd
import numpy as np
from audio_recorder_streamlit import audio_recorder
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

user_id = 'shariq'
sample_rate = 44100

def fetch_audio(text):
    response = requests.post(f"{BASE_URL}/backend/generateAUD", json={"text": text})
    if response.status_code == 200:
        return response.content
    else:
        st.write("Failed to fetch audio")
        return None
    
# Function to play audio in browser
def play_audio(audio_data):
    if audio_data:
        save_wav_file("output.wav", audio_data)
        # Display audio player in Streamlit
        audio_bytes = open('output.wav', 'rb').read()
        st.audio(audio_bytes, format='audio/wav')

@st.cache_data
def convert_pdf_to_txt_file(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()

    device.close()
    retstr.close()
    return t 

# Audio Recorder Integration
def save_wav_file2(filename, audio_data):
    with open(filename, 'wb') as f:
        f.write(audio_data)

# Custom CSS for styling (same as before)
st.markdown("""
    <style>
    body {
        background-color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF7500;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stTextInput>div>input {
        background-color: white;
        color: #000000;
        border: 2px solid #FF7500;
        border-radius: 5px;
    }
    .user-message {
        background-color: #FF7500;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: right;
        margin-bottom: 10px;
        clear: both;`
        color: #FFFFFF;
    }
    .ai-message {
        background-color: #4F6D7A;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: left;
        margin-bottom: 10px;
        clear: both;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

#st.title("Welcome to AI Powered Sales Coach")


#st.title("Welcome to AI Powered Sales Coach")

if 'start' not in st.session_state:
    st.session_state['start'] = False

st.title('Welcome to AI Powered Sales Coach')
# Step 1: Drop-down Selector (same as before)
NEW_INDEX = "(New index)"
options = get_index_list().names() + [NEW_INDEX]
option = st.selectbox("Select an option", options)

# Upload file and handle file upload logic (same as before)
if option == NEW_INDEX:
    otherOption = st.text_input("Enter your other option...")
    if otherOption:
        selection = otherOption
        index_init(otherOption, 1536)
        st.info(f":white_check_mark: New index {otherOption} created!")

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        if st.button("Generate Responses"):
            name = uploaded_file.name
            raw_text = convert_pdf_to_txt_file(uploaded_file)
            response = requests.post(f"{BASE_URL}/backend/createQuestionAnswer", json={"index": otherOption, "text": raw_text})
            st.write(f"File upload status: {response.status_code}")
    option = otherOption

st.write(f"Selected option: {option}")

# Start Test Button (same as before)
if st.button('Start Test'):
    print("fuck")
    st.session_state['start'] = True

if st.session_state['start']:
    if option:
        response = requests.post(f"{BASE_URL}/backend/fetch_questions/{option}")
        if response.json()['message'] == "Ok":

            response = requests.post(f"{BASE_URL}/backend/fetch_chats", json={'index': option, 'user_id': 'shariq'}).json()
            st.session_state['chat_history'] = response['chat']
            st.write("### Audio Recorder")
            recorder_audio = audio_recorder(text="Click to Record / Stop")

            if recorder_audio:  # If audio is captured
                audio_path = "reply2.wav"
                save_wav_file2(audio_path, recorder_audio)
                st.write("Audio recording saved!")

                response = requests.post(f"{BASE_URL}/backend/stopRecording",json={'index': option})

                if response.status_code == 200:
                    result = response.json()
                    st.session_state['chat_history'] = result['chat']
                else:
                    st.write("Failed to process recording.")

            # Display chat history (same as before)
            if st.session_state['chat_history']:
                lst = len(st.session_state['chat_history']) - 1
                audio_data = fetch_audio(st.session_state['chat_history'][lst]['message'])
                play_audio(audio_data)
                for chat in st.session_state['chat_history']:
                    if chat["type"] == "user":
                        st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ai-message">{chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.write("Please upload a file to fetch relevant question")
