from fastapi import FastAPI, UploadFile, File, Form , status, Request
from fastapi.responses import FileResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import scipy.io.wavfile as wavfile  # To save audio in .wav format
from openai import OpenAI
from util import *
import torch
import whisper
import pyaudio
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import queue
load_dotenv()

app = FastAPI()
questions = []
chats = []
qn = 0 
client  = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
p = pyaudio.PyAudio()
# Parameters for recording
sample_rate = 44100
channels = 1
audio_queue = queue.Queue()
stream = None

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getQuestions():
    ques= [
    "What was the primary technology stack used by Shariq during his AI Engineer internship at OpenInApp, and how did he optimize Instagram Reels processing?",
    "How did Shariq contribute to reducing the production time of the TTS system during his internship, and which frameworks and technologies were involved?",
    "Can you explain the key components and technologies used in Shariq's 'Intelligent SOP Generator' project, and how it improved the efficiency of SOP creation?",
    "What role did GANs and Transformers play in Shariq's research on Adversarial Bot Detection at the University of New South Wales, and how did he improve convergence by 20%?",
    "What methods and tools did Shariq employ to enhance the accuracy and reduce false positives in the Advanced Duplicate Question Detection System?"
    ]
    answers = [
        "Shariq used a technology stack that included Python, TensorFlow, and AWS during his AI Engineer internship at OpenInApp. He optimized Instagram Reels processing by implementing efficient algorithms that reduced video processing time by 30%.",
        "During his internship, Shariq contributed by streamlining the production pipeline of the TTS system, incorporating deep learning frameworks like Keras and leveraging GPU resources to achieve a 40% reduction in production time.",
        "The 'Intelligent SOP Generator' project utilized NLP techniques, Python, and machine learning algorithms. It improved SOP creation efficiency by automating data extraction and text generation, reducing manual efforts by 50%.",
        "In his research on Adversarial Bot Detection, Shariq employed GANs for generating synthetic training data and Transformers for improving model accuracy. He achieved a 20% increase in convergence speed by fine-tuning hyperparameters.",
        "Shariq utilized a combination of statistical analysis and machine learning models, including SVM and Random Forest, to enhance accuracy in the Advanced Duplicate Question Detection System, successfully reducing false positives by 15%."
    ]
    return ques

stream_p = p.open(format=pyaudio.paInt16,  # Format: 16-bit PCM (Pulse Code Modulation)
                channels=1,              # Channels: 1 (Mono)
                rate=24000,              # Sample rate: 24,000 Hz (samples per second)
                output=True)  
def speak(text):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",                   # Specify the TTS model to use
        voice="alloy",                   # Specify the voice to use for TTS
        input=text,  # Input text to be converted to speech
        response_format="pcm"            # Response format: PCM (Pulse Code Modulation)
    ) as response:
        # Iterate over the audio chunks in the response
        for chunk in response.iter_bytes(1024):  # Read 1024 bytes at a time
            stream_p.write(chunk)


def transcribe():
    print('i am in transcribe function..')
    audio_file= open("reply.wav", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    print(transcription.text)
    return transcription.text

def addToChat(type , message):
    global chats
    entry = {
        'message':message,
        'type' :type
    }
    chats.append(entry)
    print()

def genReply(text,qnum,index):
    global questions

    context = find_match(text,index)
    question = questions[qnum]
    message = f"Context:\n {context} \n\n query:\n{question}"
    sysOutput = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are an AI assisstant. your task is to understand the user query and also the context shared along with the query .Once done that you need to answer the query in relevence to the  . please keep the answer in a paragraph format such that a text to speech model can later on read it out.try to keep the response in not more than 50 words please."},
            {
                "role": "user",
                "content": message
            }
        ]
    )

    message = f"User Message:\n {text} \n\n AI Message:\n{sysOutput.choices[0].message.content} "
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are an AI assisstant. you are provided two answer to the same question, one is the user answer and the other is the AU generated answer your task is to analyse both the answers and compare them and comment on the user answer that how they can improve there answers to get closer the the one answer generated using AI "},
            {
                "role": "user",
                "content": message
            }
        ]
    )
    result = result.choices[0].message.content
    if(qnum+1) == len(questions):
        return result
    else :
        return result + f"That was the feed back I had . now your next question is \n {questions[qn+1]}"

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error during recording: {status}")
    audio_queue.put(indata.copy())

def start_recording():
    global stream, audio_queue
    audio_queue.queue.clear()  # Clear the queue before starting
    stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback)
    stream.start()
    print("Recording started...")

def stop_recording():
    global stream
    print('I am in stop recording..')
    if stream is not None:
        print('i am here to stop stream')
        stream.stop()
        stream.close()
        stream = None
        print("Recording stopped.")
        
        # Save audio data
        audio_data = np.concatenate(list(audio_queue.queue))
        audio_data = np.int16(audio_data * 32767)  # Convert float32 [-1, 1] to int16
        wavfile.write('reply.wav', 44100, audio_data)
        text = transcribe()
        # speak(text)
        return text
    else:
        return "No recording in progress."

@app.post("/fetch_questions/{index}")
async def receive_data(index: str):
    global questions
    questions = getQuestions()
    return {'status_code': status.HTTP_200_OK}

@app.get("/fetch_chats")
async def receive_data(req: Request):
    global questions
    global chats
    global qn
    contents = await req.json()
    # chat = fetchChat()
    if chats == []:
        message = f'Hello i am Siva your AI proctor, Are you ready!! first question for you is, {questions[0]}'
        speak(message)
        chats = [{"message": message, "type":"AI"}]
    return {'status_code': status.HTTP_200_OK, 'chat':chats}


@app.post("/stopRecording")
async def upload_file(req: Request):
    global qn
    text = stop_recording()
    contents = await req.json()
    qn = qn+1
    index = contents['index']
    addToChat(type = 'User',message = text)
    reply = genReply(text,qn,index)
    speak(reply)
    addToChat(type = 'AI',message = reply)
    return {'status_code':status.HTTP_200_OK,'chat':chats}



@app.post("/start-recording")
def start_recording_endpoint():
    start_recording()
    return {"message": "Recording started"}

# @app.post("/upload_file")
# def start_recording_endpoint():
#     start_recording()
#     return {"message": "Recording started"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
