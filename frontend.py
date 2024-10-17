import streamlit as st
import requests
from util import *
import queue
import sounddevice as sd
import numpy as np


user_id = 'shariq'
sample_rate = 44100
# audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    global audio_queue
    if status:
        print(status)
    audio_queue.put(indata.copy())

def start_recording():
    print("Recording started...")
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
    stream.start()
    return stream

def stop_recording(stream):
    print("Recording stopped.")
    stream.stop()
    stream.close()

st.markdown("""
    <style>
    .user-message {
        background-color: #d1e7dd;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: right;
        margin-bottom: 10px;
        clear: both;
    }
    .ai-message {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: left;
        margin-bottom: 10px;
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)

if 'start' not in st.session_state:
    st.session_state['start'] = False
if 'stream' not in st.session_state:
    st.session_state['stream'] = None


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
            # Send file to dummy API
            files = {'file': uploaded_file}
            # response = requests.post("https://0.0.0.0:8000/upload_file", files=files,json={"Index": option})
            # st.write(f"File upload status: {response.status_code}")
    option = otherOption
st.write(f"Selected option: {option}")

if st.button('Start Test'):
    st.session_state['start']=True


if st.session_state['start']:

    if option:
        requests.post(f"http://0.0.0.0:8000/fetch_questions/{option}")
    
    response = requests.get(f"http://0.0.0.0:8000/fetch_chats/",json={'index':option,'user_id':'shariq'}).json()
    st.session_state['chat_history'] = response['chat']
    # st.success("Chat history fetched successfully!")

    if st.button("Start Recording"):
        response = requests.post("http://0.0.0.0:8000//start-recording")
        if response.status_code == 200:
            st.write("Recording started!")
        else:
            st.write("Failed to start recording.")


    if st.button("Stop Recording"):
        
        response = requests.post(f"http://0.0.0.0:8000/stopRecording/",json={'index':option})
        # response = response.json()
        if response.status_code == 200:
            result = response.json()
            print(result)
            # st.write(result['message'])
            st.session_state['chat_history'] = result['chat']
        else:
            st.write("Failed to stop recording.")


    # Display chat history with alignment
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            if chat["type"] == "user":
                st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{chat["message"]}</div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .user-message {
        background-color: #d1e7dd;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: right;
        margin-bottom: 10px;
        clear: both;
    }
    .ai-message {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 8px;
        text-align: left;
        max-width: 60%;
        float: left;
        margin-bottom: 10px;
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)



