from fastapi import FastAPI, Request ,Response
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os, json
load_dotenv()
import pyaudio



client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
p = pyaudio.PyAudio()
app = FastAPI()

stream_p = p.open(format=pyaudio.paInt16,  # Format: 16-bit PCM (Pulse Code Modulation)
                channels=1,              # Channels: 1 (Mono)
                rate=24000,              # Sample rate: 24,000 Hz (samples per second)
                output=True)  
def speak(text):
    audio_data = b''  # to store audio chunks
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm"
    ) as response:
        for chunk in response.iter_bytes(1024):  # Read 1024 bytes at a time
            audio_data += chunk  # append each chunk to the audio_data

    return audio_data


@app.post("/generateAUD")
async def stop_recording_endpoint(req : Request):
    content = await req.json()
    text = content['text']
    audio_data = speak(text)
    headers = {
        'Content-Disposition': 'inline; filename="response.wav"',
        'Content-Type': 'audio/wav'
    }

    return Response(content=audio_data, media_type="audio/wav", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)