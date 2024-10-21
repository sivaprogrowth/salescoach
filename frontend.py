import streamlit as st
import requests
from util import *
import queue , wave
import sounddevice as sd
import numpy as np
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


user_id = 'shariq'
sample_rate = 44100

def audio_callback(indata, frames, time, status):
    global audio_queue
    if status:
        print(status)
    audio_queue.put(indata.copy())

def fetch_audio(text):
    response = requests.post(f"http://0.0.0.0:8000/generateAUD", json={"text": text})
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

def save_wav_file(filename, audio_data, sample_rate=22050, num_channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # assuming 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def start_recording():
    print("Recording started...")
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
    stream.start()
    return stream

def stop_recording(stream):
    print("Recording stopped.")
    stream.stop()
    stream.close()

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

# Custom CSS for styling
st.markdown("""
    <style>
    /* Set background color */
    body {
        background-color: #FFFFFF;
    }
    /* Customize primary color elements */
    .stButton>button {
        background-color: #FF7500;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stTextInput>div>input {
        background-color: white;
        color: #000000; /* Input text color */
        border: 2px solid #FF7500; /* Input border color */
        border-radius: 5px;
    }
    /* User and AI message bubbles */
    .user-message {
        background-color: #FF7500;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: right;
        margin-bottom: 10px;
        clear: both;
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

st.title("Welcome to AI Powered Sales Coach")


if 'start' not in st.session_state:
    st.session_state['start'] = False
if 'stream' not in st.session_state:
    st.session_state['stream'] = None

# Step 1: Drop-down Selector
NEW_INDEX = "(New index)"
options = get_index_list().names() + [NEW_INDEX]
option = st.selectbox("Select an option", options)

if option == NEW_INDEX:
    otherOption = st.text_input("Enter your other option...")
    if otherOption:
        selection = otherOption
        index_init(otherOption, 1536)
        st.info(f":white_check_mark: New index {otherOption} created! ")

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        if st.button("Send File"):
            name = uploaded_file.name
            raw_text = convert_pdf_to_txt_file(uploaded_file)
            response = requests.post("http://localhost:8000/createQuestionAnswer",json={"index": otherOption, "text": raw_text})
            st.write(f"File upload status: {response.status_code}")
    option = otherOption
st.write(f"Selected option: {option}")

if st.button('Start Test'):
    st.session_state['start']=True

if st.session_state['start']:

    if option:
        requests.post(f"http://0.0.0.0:8000/fetch_questions/{option}")
    
    response = requests.get(f"http://0.0.0.0:8000/fetch_chats/",json={'index':option,'user_id':'shariq'}).json()
    st.session_state['chat_history'] = response['chat']

    if len(st.session_state['chat_history']) == 1:
        audio_data = fetch_audio(st.session_state['chat_history'][0]['message'])
        play_audio(audio_data)

    if st.button("Start Recording"):
        response = requests.post("http://0.0.0.0:8000//start-recording")
        if response.status_code == 200:
            st.write("Recording started!")
        else:
            st.write("Failed to start recording.")

    if st.button("Stop Recording"):
        response = requests.post(f"http://0.0.0.0:8000/stopRecording/",json={'index':option})
        if response.status_code == 200:
            result = response.json()
            st.session_state['chat_history'] = result['chat']
        else:
            st.write("Failed to stop recording.")

    # Display chat history with alignment
    if st.session_state['chat_history']:
        lst = len(st.session_state['chat_history'])-1
        audio_data = fetch_audio(st.session_state['chat_history'][lst]['message'])
        play_audio(audio_data)
        for chat in st.session_state['chat_history']:
            if chat["type"] == "user":
                st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{chat["message"]}</div>', unsafe_allow_html=True)
