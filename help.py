import streamlit as st
import wave

import requests

# Fetch audio data from backend
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

# UI to get input text and play audio
st.title("Text-to-Speech Audio Player")

user_input = st.text_input("Enter text to convert to speech", "")
if st.button("Generate and Play Audio"):
    if user_input:
        audio_data = fetch_audio(user_input)
        play_audio(audio_data)
    else:
        st.write("Please enter some text.")
