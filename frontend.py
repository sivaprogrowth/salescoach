import streamlit as st
import requests
from serverdb import get_index_list_glific
from util import *
import os
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")

user_id = 'shariq'

def fetch_audio(text,lang):
    response = requests.post(f"{BASE_URL}/backend/generateAUD", json={"text": text,"lang":lang})
    if response.status_code == 200:
        return response.content
    else:
        st.write("Failed to fetch audio")
        return None

def play_audio(audio_data):
    if audio_data:
        save_wav_file("output.wav", audio_data)
        audio_bytes = open("output.wav", "rb").read()
        st.audio(audio_bytes, format="audio/wav")

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

# Initialize session state variables if not present
if "start" not in st.session_state:
    st.session_state["start"] = False
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = None
if "index_initialized" not in st.session_state:
    st.session_state["index_initialized"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "language" not in st.session_state:
    st.session_state["language"] = "English"

st.title("Welcome to AI Powered Sales Coach")
# Add language selection dropdown
st.sidebar.title("Select Input Medium")
language_options = ["English", "Hindi", "Telugu"]
selected_language = st.sidebar.selectbox("Select Language", language_options)

if selected_language:
    # Store selected language in session state
    st.session_state["language"] = selected_language

# Step 1: Drop-down Selector
NEW_INDEX = "(New index)"
options = get_index_list_glific() + [NEW_INDEX]

option = st.selectbox("Select an option", options, index=options.index(st.session_state["selected_option"]) if st.session_state["selected_option"] in options else 0)

if option == NEW_INDEX:
    otherOption = st.text_input("Name the new File")
    if otherOption:
        st.session_state["selected_option"] = otherOption  # Store selection
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file  # Store the file
            st.write("File uploaded successfully!")
            st.info(f":white_check_mark: New index {otherOption} created!")
            st.info(f":white_check_mark: Now Creating QnA")
            name = st.session_state["uploaded_file"].name
            raw_text = extract_text_from_pdf(st.session_state["uploaded_file"].read())
            response = requests.post(f"{BASE_URL}/backend/createQuestionAnswer", json={"index": otherOption, "text": raw_text,"lang":st.session["language"]})
            st.write(f"File upload status: {response.status_code}")

else:
    st.session_state["selected_option"] = option

st.write(f"Selected option: {st.session_state['selected_option']}")

# Start Test Button
if st.button("Start Test"):
    st.session_state["start"] = True

if st.session_state["start"]:
    if st.session_state["selected_option"]:
        response = requests.post(f"{BASE_URL}/backend/fetch_questions/{st.session_state['selected_option']}")
        if response.json().get("message") == "Ok":
            chat_response = requests.post(f"{BASE_URL}/backend/fetch_chats", json={"index": st.session_state["selected_option"], "user_id": user_id}).json()
            st.session_state["chat_history"] = chat_response["chat"]

            st.write("### Audio Recorder")
            recorder_audio = audio_recorder(text="Click to Record / Stop")

            if recorder_audio:
                audio_path = "reply2.wav"
                with open(audio_path, "wb") as f:
                    f.write(recorder_audio)
                st.write("Audio recording saved!")

                response = requests.post(f"{BASE_URL}/backend/stopRecording", json={"index": st.session_state["selected_option"]})

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["chat_history"] = result["chat"]
                else:
                    st.write("Failed to process recording.")

            if st.session_state["chat_history"]:
                lst = len(st.session_state["chat_history"]) - 1
                audio_data = fetch_audio(st.session_state["chat_history"][lst]["message"],st.session["language"])
                play_audio(audio_data)

                for chat in st.session_state["chat_history"]:
                    if chat["type"] == "user":
                        st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ai-message">{chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.write("Please upload a file to fetch relevant questions.")
