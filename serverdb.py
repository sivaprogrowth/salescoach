from fastapi import FastAPI, UploadFile, File, Form , status, Request , Response, HTTPException, Query
from fastapi.responses import FileResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from util import *
from service import *
from datetime import datetime
from pinecone import Pinecone
from botocore.exceptions import BotoCoreError, ClientError
from pydub import AudioSegment
from dotenv import load_dotenv
import scipy.io.wavfile as wavfile 
import numpy as np
import sounddevice as sd
import pyaudio
import os , json
import mysql.connector
import queue
import requests
import io
from io import BytesIO
from typing import Optional, List
import re
import boto3
import tempfile
import crud
import shutil
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
USER = os.getenv('DB_USER')
HOST = os.getenv('HOST')
connection = mysql.connector.connect(
    user = USER,
    host = HOST,
    database = DATABASE,
    passwd = PASSWD
)

sns_client = boto3.client('sns', region_name='ap-south-1')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN') 
SNS_TOPIC_TEXT_RAG_ARN = os.getenv('SNS_TOPIC_TEXT_RAG_ARN')
# Glific API credentials
GLIFIC_PHONE_NUMBER = os.getenv("GLIFIC_PHONE_NUMBER")
GLIFIC_PASSWORD = os.getenv("GLIFIC_PASSWORD")
GLIFIC_AUTH_URL = os.getenv("GLIFIC_AUTH_URL")
GLIFIC_SEND_URL = os.getenv("GLIFIC_SEND_URL")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
LESSON_BUCKET = os.getenv("LESSON_BUCKET")
pinecone_database_name = os.getenv("PINECONE_DATABASE_NAME")


cursor = connection.cursor()

app = FastAPI()
questions = []
answers = []
qn = 0 
client  = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
p = pyaudio.PyAudio()
sample_rate = 44100
channels = 1
audio_queue = queue.Queue()
stream = None

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getQnA(text):

    messages = [
        {
            "role": "system",
            "content": """
            You are an AI that excels at analyzing complex text and generating high-quality, knowledge-assessment questions and their respective answers. 
            Your task is to evaluate the provided text and generate 5 relevant questions that assess a person's understanding of key points, insights, or concepts from the material. 
            Each question should test knowledge of significant information, and you should also provide the best possible answer for each question.

            Please format the response as a strict list of dictionaries in the following JSON format:

            [
                {
                    "Question": ".....",
                    "Answer": "......"
                },
                {
                    "Question": ".....",
                    "Answer": "......"
                },
                {
                    "Question": ".....",
                    "Answer": "......"
                }
            ]

            Ensure that the output strictly adheres to this format so it can be converted into JSON and stored in a database.
            """
        },
        {
            "role": "user",
            "content": f"Here is the text: {text}"
        }
    ]

    qna = client.chat.completions.create(
        model="gpt-4o", 
        messages=messages,
        max_tokens=1000,
        temperature=0.7, 
        n=1,
        stop=None
    )
    qna = qna.choices[0].message.content
    return qna

def saveQnA(text , index):
    text = text.replace('```json', '').replace('```', '')
    print(text)
    qnas = json.loads(text)
    print (qnas)
    for idx , qna in enumerate(qnas):
        current_datetime = datetime.now()
        timestamp = int(current_datetime.timestamp())
        data = (
            timestamp,
            qna['Question'],   # question
            qna['Answer'],  # answer
            index # index
        )
        insert_query = """
                            INSERT INTO qna (id, question, answer, idx) 
                            VALUES (%s, %s, %s, %s)
                        """
        cursor.execute(insert_query, data)
        connection.commit()

def getQuestionsAnswers(index):
    cursor.execute(f"select question , answer from qna where idx = '{index}'")
    result = cursor.fetchall()
    ques = []
    ans = []
    for qna in result:
        ques.append(qna[0])
        ans.append(qna[1])
    return ques , ans

def addToChat(type , message , index):
    data = (
            0,
            type,   
            message, 
            index, 
            1
        )
    insert_query = """
                    INSERT INTO chats (id, message_type, message_content, `index`, user_id) 
                    VALUES (%s, %s, %s, %s, %s)
                """
    cursor.execute(insert_query, data)
    connection.commit()

def genReply(text,qnum,index):
    global questions
    global answers
    answer = answers[qnum]

    message = f"User Message:\n {text} \n\n AI Message:{answer}"
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an expert sales coach. You have experience in evaluating sales pitches and responses of salespersons and give them quality and unbiased focused feedback. You can also rate the responses from the salespersons with the ideal responses.
 For evaluation,you are provided two responses to the same question, one is the response from the salesperson and the other is the AI generated answer. Your task is to analyse both the answers and compare them and provide the following details
Rate the response from the salesperson against the ideal response for the question. Give your rating on a scale of 5 where 5 is the best and 1 is the worst.
Give 3 bulleted comments to the salesperson on how they can improve there answers to get closer the the one answer generated using AI
Give 4 action items that the salesperson has to follow for the next 1 month to improve their sales pitch
It is compulsory for you to rate the salesperson response. """},
            {
                "role": "user",
                "content": message
            }
        ]
    )
    result = result.choices[0].message.content
    if(qnum+1) == len(questions):
        return result + f"That was the feed back I had. Thanks for taking the test."
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
        return text
    else:
        return "No recording in progress."
    
def fetchChat(index):
    cursor.execute(f"select message_type, message_content from chats where `index` = '{index}'")
    result = cursor.fetchall()
    chats = []
    if len(result)>0:
        for chat in result:
            entry = {'type':chat[0],'message':chat[1]}
            chats.append(entry)
    return chats

async def run_script(message: str, idx : str) -> str:
    try:
        print('Message received from user:', message)
        
        # Process the message and generate an embedding
        response = client.embeddings.create(
            input=[message],
            model="text-embedding-ada-002",
        )
        embedding = response.data[0].embedding
        print('Embeddings created')
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(idx)
        # Send embedding to Pinecone and perform a query
        pinecone_response = index.query(vector=embedding, top_k=1,include_metadata=True,)
        pinecone_result = "\n".join([match['metadata']['text'] for match in pinecone_response['matches']])
        print('Pinecone query completed')
        
        # Prepare prompt for OpenAI completion
        prompt = f"""Act as a guide for people with strong coaching and career counselling credentials and 
                    who has lot of experience of guiding young employees and candidates in India.
                    Based on the following data:\n{pinecone_result}\nGenerate a response to the query message {message}. If you dont know the answer, give them a message “Currently I dont know the response but please reach me on https://www.yogyabano.com/contact-us to get specific response for your query”.Be specific and provide your responses in 5- 8 bullet points. Well formatted and with relevant examples, wherever required.
                    In the end of every response, please give them a relevant sentence that prompts them to ask more questions.
                    """
        
        # prompt = f"Based on the following data:\n{pinecone_result}\nGenerate a response to the query message {message}."
        
        # Get response from OpenAI's completion model
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        print("OpenAI response received")
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error in run_script: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

# Function to get the auth token from the Glific authentication API
def get_auth_token():
    try:
        response = requests.post(
            GLIFIC_AUTH_URL,
            json={
                "user": {
                    "phone": GLIFIC_PHONE_NUMBER,
                    "password": GLIFIC_PASSWORD,
                }
            }
        )
        response.raise_for_status()
        auth_token = response.json().get("data", {}).get("access_token")
        if not auth_token:
            raise HTTPException(status_code=500, detail="Auth token not received")
        print("Auth token received:", auth_token)
        return auth_token
    except requests.HTTPError as error:
        print("Error fetching auth token:", error)
        raise HTTPException(status_code=500, detail="Unable to fetch auth token")
        print("Error fetching auth token:", error)
        raise HTTPException(status_code=500, detail="Unable to fetch auth token")


async def send_to_glific_api(flow_id: int, contact_id: int, result: str):
    try:
        auth_token = get_auth_token()
        graphql_query = """
        mutation resumeContactFlow($flowId: ID!, $contactId: ID!, $result: Json!) {
          resumeContactFlow(flowId: $flowId, contactId: $contactId, result: $result) {
            success
            errors {
                key
                message
            }
          }
        }
        """
        graphql_variables = {
            "flowId":flow_id,
            "contactId":contact_id,
            "result":f"{{\"result\":{{\"message\":\"{result}\"}}}}"
        }
        headers = {
            "authorization": f"{auth_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            GLIFIC_SEND_URL,
            json={
                "query": graphql_query,
                "variables": graphql_variables
            },
            headers=headers
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending data to Glific API: {e}")

# Utility Function: Validate Email
def is_valid_email(email: str) -> bool:
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(email_regex, email))

# Utility Function: Fetch User by Email
def get_user_by_email(email: str):
    cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
    return cursor.fetchone()

# Utility Function: Add User to Database
def add_user(email: str, password: str):
    cursor.execute("INSERT INTO login (email, password) VALUES (%s, %s)", (email, password))
    connection.commit()
    
@app.post("/backend/generateAUD")
async def stop_recording_endpoint(req : Request):
    content = await req.json()
    text = content['text']
    audio_data = speak(text)
    headers = {
        'Content-Disposition': 'inline; filename="response.wav"',
        'Content-Type': 'audio/wav'
    }

    return Response(content=audio_data, media_type="audio/wav", headers=headers)

@app.post("/backend/fetch_questions/{index}")
async def receive_data(index: str):
    global questions
    global answers
    print(index)
    questions , answers = getQuestionsAnswers(index)
    if(len(questions)==0):
        message  = "Please Upload a file to fetch Questions."
    else:
        message  = "Ok"
        print(questions)
    return {'status_code': status.HTTP_200_OK, 'message':message}

@app.post("/backend/fetch_chats")
async def receive_data(req: Request):
    global questions
    global qn
    print("json = ",req.json)
    contents = await req.json()
    print("index is ",contents['index'])
    chats = fetchChat(contents['index'])
    print("first chat fetch done")
    if len(chats) == 0:
        message = f'Hello i am Siva your AI proctor, Are you ready!! first question for you is, {questions[0]}'
        addToChat(type = 'AI',message = message,index = contents['index'])
        chats = fetchChat(contents['index'])
    return {'status_code': status.HTTP_200_OK, 'chat':chats}


@app.post("/backend/stopRecording")
async def upload_file(req: Request):
    global qn
    text = transcribe()
    contents = await req.json()
    qn = qn+1
    index = contents['index']
    addToChat(type = 'User',message = text , index=index)
    reply = genReply(text,qn,index)
    addToChat(type = 'AI',message = reply,index = index)
    chats = fetchChat(index)
    return {'status_code':status.HTTP_200_OK,'chat':chats}



@app.post("/backend/start-recording")
def start_recording_endpoint():
    start_recording()
    return {"message": "Recording started"}

@app.post("/backend/createQuestionAnswer")
async def stop_recording_endpoint(req : Request):
    print(req)
    content = await req.json()
    print(content)
    index = content['index']
    text = content['text']
    print(index)
    print(text)
    qna = getQnA(text)
    saveQnA(qna,index)
    return{'status_code':status.HTTP_200_OK}

@app.post("/backend/publish")
async def publish_to_sns(request: Request):
    try:
        # Publish message to SNS with attributes
        print(SNS_TOPIC_ARN)
        request = await request.json()
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=request['message'],
            MessageAttributes={
                'contactId': {
                    'DataType': 'Number',
                    'StringValue': str(request['contactId'])
                },
                'flowId': {
                    'DataType': 'Number',
                    'StringValue': str(request['flowId'])
                },
                 'idx': {
                    'DataType': 'String',
                    'StringValue': request['idx']
                }
            }
        )
        print(f"Message sent to SNS, MessageId: {sns_response['MessageId']}")
        return {"status": "Message sent to SNS successfully", "MessageId": sns_response['MessageId']}

    except (BotoCoreError, ClientError) as error:
        print(f"Failed to publish message to SNS: {error}")
        raise HTTPException(status_code=500, detail="Failed to send message to SNS")
    


@app.post("/backend/sns")
async def sns_listener(request: Request):
    headers = request.headers
    message_type = headers.get("x-amz-sns-message-type")

    if not message_type:
        raise HTTPException(status_code=400, detail="Invalid message type header")

    body = await request.json()

    # Check if it's a SubscriptionConfirmation message
    if message_type == "SubscriptionConfirmation" and body['SubscribeURL']:
        # Confirm the subscription
        response = requests.get(body['SubscribeURL'])
        if response.status_code == 200:
            print("Subscription confirmed.")
        else:
            print("Failed to confirm subscription.")
        return {"status": "Subscription confirmation handled"}

    # Handle Notification message
    elif message_type == "Notification":
        print(f"Received notification: {body['Message']}")
        # Retrieve contactId and flowId from MessageAttributes
        message_attributes = body['MessageAttributes'] or {}
        contact_id = message_attributes.get('contactId', {}).get('Value')
        flow_id = message_attributes.get('flowId', {}).get('Value')
        idx = message_attributes.get('idx', {}).get('Value')
        print(flow_id)
        print(contact_id)
        result  = await run_script(body['Message'],idx)
        result = repr(result).strip("'\"")
        #result_formatted = repr(result).strip("'").replace("\\", "\\\\")
        response =  await send_to_glific_api(flow_id , contact_id,result)
        print("sent to glific successfully: ",response)
    # If unknown message type, return 400 error
    else:
        raise HTTPException(status_code=400, detail="Unknown message type")

@app.post("/backend/fetchChatVoice")
async def fetchChatVoice(req: Request):

    print("Fetching chat voice...")
    content = await req.json()
    option = content['option']
    print(f"Option received: {option}")
    response = requests.post(f"https://salescoach.yogyabano.com/backend/fetch_chats", json={'index': option, 'user_id': 'shariq'}).json()

    if 'chat' not in response:
        print("Error: 'chat' key not found in the response.")
        return {'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR, 'message': 'Error fetching chat'}
    
    chats = response['chat']    
    length = len(chats) - 1
    audio_data = chats[length]['message']    
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    file_name = f"output_{datetime_string}.wav"
    print(f"Saving audio data to {file_name}...")

    save_wav_file(file_name, audio_data)
    audio_url = upload_audio_to_s3(file_name)
    print(f"Audio file uploaded to S3. URL: {audio_url}")
    
    # Remove the local file after uploading
    print(f"Removing local file {file_name}...")
    os.remove(file_name)
    return {'status_code': status.HTTP_200_OK, 'audio_url': audio_url, 'text':audio_data} 

@app.post("/backend/stopRecordingVoice")
async def fetchChatVoice(req: Request):
    content  = await req.json()
    url = content['url']
    download_audio(url,"reply2.wav", "wav")
    text = transcribe()
    qn = qn+1
    index = content['index']
    addToChat(type = 'User',message = text , index=index)
    reply = genReply(text,qn,index)
    addToChat(type = 'AI',message = reply,index = index)
    return {'status_code':status.HTTP_200_OK}

@app.post("/backend/clearChat")
async def fetchChatVoice(req: Request):
    content  = await req.json()
    index = content['index']
    user_id = 1
    cursor.execute(f"DELETE FROM chats WHERE `index` = {index}")
    return {'status_code':status.HTTP_200_OK}

@app.post("/backend/publishTextRag",status_code=status.HTTP_200_OK)
async def publish_to_sns_text_rag(request: Request):
    try:
        # Publish message to SNS with attributes
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
        request = await request.json()

        course_number = int(request['course_number'])
        courses_name = request['courses_name']
        course_title = courses_name.split("\n")[course_number].split(".")[1].strip()

        lesson_number = int(request['lesson_number'])
        lessons_name = request['lessons_name']
        lesson_name = lessons_name.split("\n")[lesson_number].split(".")[1].strip()

        print("course title ",course_title)
        print("lesson name ",lesson_name)
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_TEXT_RAG_ARN,
            Message=request['message'],
            MessageAttributes={
                'transactionId': {
                    'DataType': 'Number',
                    'StringValue': datetime_string
                },
                 'lesson_name': {
                    'DataType': 'String',
                    'StringValue': lesson_name
                },
                'user_id': {
                    'DataType': 'String',
                    'StringValue': request['user_id']
                },
                'course_title': {
                    'DataType': 'String',
                    'StringValue': course_title
                },
                'flow_id': {
                    'DataType': 'Number',
                    'StringValue': str(request['flow_id'])
                },
                'contact_id': {
                    'DataType': 'Number',
                    'StringValue': str(request['contact_id'])
                }
            }
        )
        print(f"Message sent to SNS, MessageId: {sns_response['MessageId']}")
        return {"status": "Message sent to SNS successfully", "MessageId": sns_response['MessageId'], "TransactionId":datetime_string}
    
    except (BotoCoreError, ClientError) as error:
        print(f"Failed to publish message to SNS: {error}")
        raise HTTPException(status_code=500, detail="Failed to send message to SNS")
    
@app.post("/backend/snsTextRag")
async def sns_listener_TextRag(request: Request):
    headers = request.headers
    message_type = headers.get("x-amz-sns-message-type")

    if not message_type:
        raise HTTPException(status_code=400, detail="Invalid message type header")
    body = await request.json()

    # Check if it's a SubscriptionConfirmation message
    if message_type == "SubscriptionConfirmation" and body['SubscribeURL']:
        # Confirm the subscription
        response = requests.get(body['SubscribeURL'])
        if response.status_code == 200:
            print("Subscription confirmed.")
        else:
            print("Failed to confirm subscription.")
        return {"status": "Subscription confirmation handled"}

    # Handle Notification message
    elif message_type == "Notification":
        print(f"Received notification: {body['Message']}")
        # Retrieve contactId and flowId from MessageAttributes
        message_attributes = body['MessageAttributes'] or {}
        transactionId = message_attributes.get('transactionId', {}).get('Value')
        lesson_name = message_attributes.get('lesson_name', {}).get('Value')
        flow_id = message_attributes.get('flow_id', {}).get('Value')
        contact_id = message_attributes.get('contact_id', {}).get('Value')
        user_id = message_attributes.get('user_id', {}).get('Value')
        course_title = message_attributes.get('course_title', {}).get('Value')
        print("course_title",course_title)
        print("lesson name ",lesson_name)
        idx = get_index_by_lesson(lesson_name)
        print("idx ",idx)
        course_id = get_course_id_by_name(course_title)
        result  = await run_script(body['Message'],idx.split(".")[-2])
        result = repr(result).strip("'\"")
        response =  await send_to_glific_api(flow_id , contact_id,result)
        print("sent to glific successfully: ",response)
        data = (
            result, # message
            transactionId,  # transactionId
            idx,
            course_id,
            user_id 
        )
        insert_query = """
                        INSERT INTO textRag (message, tranId, idx, course_id, user_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """
        cursor.execute(insert_query, data)
        connection.commit()
    # If unknown message type, return 400 error
    else:
        raise HTTPException(status_code=400, detail="Unknown message type")
    
@app.post("/backend/retriever")
def getMessage(req :Request):

    tranId = req.json()['transactionId']
    cursor.execute(f"select message from textRag where tranId = '{tranId}'")
    result = cursor.fetchall()
    if result != []:
        return {"status_code": status.HTTP_200_OK, "Message": result[0][0]}
    else:
        print(f"Process in wait ")
        return {"status_code": status.HTTP_202_OK}

@app.post("/backend/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), idx: int = Form(...), file_name: str = Form(...)):
    # Create a temporary file
    if file is None:
        print("no file")
        raise HTTPException(status_code=400, detail="File not provided.")
    print("all okh")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_name = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
    
    try:
        # Open the temporary file in binary read mode and pass the file object
        with open(temp_file_name, "rb") as pdf_file:
            text = convert_pdf_to_txt_file(pdf_file)
        
        # Upload the extracted text to Pinecone
        upload_file_to_pinecone(text, file_name, idx)
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_name)
    
    return {"filename": file.filename, "content": text}


@app.post("/backend/login")
async def login(request: Request):
    data = await request.json()
    email = data.get("email")
    password = data.get("password")

    # Validate email format
    if not is_valid_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format."
        )

    # Fetch user by email
    user = get_user_by_email(email)

    if user:
        # Check if password matches
        if user[1] == password:  # Assuming `password` is the second column in `users`
            return {"status_code": status.HTTP_200_OK, "message": "Login successful"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password."
            )
    else:
        # Add new user to the database
        add_user(email, password)
        return {"status_code": status.HTTP_200_OK, "message": "User created successfully"}
    
@app.post("/backend/courses", status_code=status.HTTP_201_CREATED)
async def create_course(req: Request):
    try:
        # Parse the incoming JSON data
        data = await req.json()
        print(data)

        # Validation checks for required fields
        required_fields = ["title", "industry", "description", "company_id"]
        missing_fields = [field for field in required_fields if field not in data or not data[field]]

        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # If validation passes, proceed to create the course
        course_id = create_courses_service(data)
        return {"message": "Course created successfully", "course_id": course_id}

    except Exception as e:
        # Catch-all for any unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/backend/getCourses", status_code=status.HTTP_200_OK)
async def get_one_course(req:Request):
    print("i am here")
    data = await req.json()
    course_id = data['course_id']
    course = get_one_course_service(course_id)
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    return course

@app.post("/backend/allCourses", status_code=status.HTTP_200_OK)
async def get_all_courses(
    req: Request ,
    date: str = Query(None, description="Filter by a specific date (format: YYYY-MM-DD)"),
    latest_added: bool = Query(None, description="Filter by the most recently added courses"),
):
    data = await req.json()
    company_id = data['company_id']
    print("i am here")
    print(company_id)
    try:
        courses = get_all_courses_service(company_id=company_id, date=date, latest_added=latest_added)

        if not courses:
            raise HTTPException(status_code=404, detail="No courses found")

        return courses
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))
    
@app.post("/backend/updateCourses", status_code=status.HTTP_200_OK)
async def update_courses(req: Request):
    data = await req.json()
    course_id = data['course_id']
    update_course_service(course_id, data)
    return {"message": "Course updated successfully"}

@app.post("/backend/deleteCourses", status_code=status.HTTP_200_OK)
async def delete_courses(req: Request):
    data = await req.json()
    course_id = data['course_id']
    delete_course_services(course_id)
    return {"message": "Course deleted successfully"}

# Lessons APIs
@app.post("/backend/lessons", status_code=status.HTTP_201_CREATED)
async def create_lesson(req:Request):
    try:
        data = await req.json()
        file_name = data["file_name"]
        # Download file directly from S3
        response = s3.get_object(Bucket=LESSON_BUCKET, Key=file_name)

        if response:
            pdf_stream = response['Body'].read()  
            # Create a file-like stream
            text = extract_text_from_pdf(BytesIO(pdf_stream))
            # Upload to Pinecone
            pdf_file_name = file_name.split('.')[-2]
            res = await upload_file_to_pinecone(text, file_name, pdf_file_name)
            print (res)

        # Create lesson in database
        lesson_id = create_lesson_service(data)
        return {"message": "Lesson created successfully", "lesson_id": lesson_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@app.post("/backend/allLessons",status_code=status.HTTP_200_OK)
async def get_lesson(req: Request):
    data = await req.json()
    course_id = data['course_id']
    try:
        lesson = get_lessons_service(course_id)
        if not lesson:
            raise HTTPException(status_code=status.HTTP_200_OK, detail="Lesson not found")
        return lesson
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))
    
@app.put("/backend/lessons",status_code=status.HTTP_200_OK)
async def update_lesson(req: Request):

    data = await req.json()
    if "file_name" in data:    
        file_name = data["file_name"]
        #Deleting previous PDF
        lesson_id = data["lesson_id"]
        prev_PDF = get_lesson_PDF(lesson_id)
        delete_index(prev_PDF)

        response = s3.get_object(Bucket=LESSON_BUCKET, Key=file_name)
        if response:
            pdf_stream = response['Body'].read()  
            # Create a file-like stream
            text = convert_pdf_to_txt_file(BytesIO(pdf_stream))
            # Upload to Pinecone
            pdf_file_name = file_name.split('.')[-2]
            upload_file_to_pinecone(text, file_name, pdf_file_name)
            
    # Filter only fields that are not None

    update_lesson_service(data)
    return {"message": "Lesson updated successfully"}

@app.post("/backend/deleteLessons",status_code=status.HTTP_200_OK)
async def delete_lesson(req: Request):
    data = await req.json()
    lesson_id = data['lesson_id']
    print(lesson_id)
    delete_lesson_service(lesson_id)
    return {"message": "Lesson deleted successfully"}

# Assessments APIs
@app.post("/backend/assessments",status_code=status.HTTP_201_CREATED)
async def create_assessment(req: Request):
    data = await req.json()
    assessment_id = create_assessment_service(data)
    return {"message": "Assessment created successfully", "assessment_id": assessment_id}

@app.post("/backend/allAssessments")
async def get_assessment(req: Request):
    data = await req.json()
    try:
        assessment = get_all_assessment_service(data["course_id"])
        if not assessment:
            raise HTTPException(status_code=status.HTTP_200_OK, detail="Assessment not found")
        return {"status_code": status.HTTP_200_OK, "data": assessment}
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))

@app.put("/backend/assessments")
async def update_assessment(req: Request):
    data = await req.json()
    update_assessment_service(data["assessment_id"], data)
    return {"status_code": status.HTTP_200_OK, "message": "Assessment updated successfully"}

@app.post("/backend/deleteAssessments")
async def delete_assessment(req: Request):
    data = await req.json()
    delete_assessment_service(data["assessment_id"])
    return {"status_code": status.HTTP_200_OK, "message": "Assessment deleted successfully"}

# Feedback APIs
@app.post("/backend/feedbacks",status_code=status.HTTP_201_CREATED)
async def create_feedback(req: Request):
    data = await req.json()
    feedback_id = create_feedback_service(data)
    return {"message": "Feedback created successfully", "feedback_id": feedback_id}

@app.post("/backend/getFeedbacks",status_code=status.HTTP_200_OK)
async def get_feedback(req: Request):
    data = await req.json()
    feedback_id = data['feedback_id']
    feedback = get_feedback_service(feedback_id)
    if not feedback:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    return feedback

@app.put("/backend/feedbacks",status_code=status.HTTP_200_OK)
async def update_feedback(req: Request):
    data = await req.json()
    feedback_id = data['feedback_id']
    update_feedback_service(feedback_id, data)
    return {"message": "Feedback updated successfully"}

@app.post("/backend/deleteFeedbacks",status_code=status.HTTP_200_OK)
async def delete_feedback(req : Request):
    data = await req.json()
    feedback_id = data['feedback_id']
    delete_feedback_service(feedback_id)
    return {"message": "Feedback deleted successfully"}

@app.post("/backend/allFeedbacks",status_code=status.HTTP_200_OK)
async def get_all_feedback(req : Request):
    data = await req.json()
    course_id = data['course_id']
    try:
        feedbacks = get_all_feedback_services(course_id)
        if not feedbacks:
                raise HTTPException(status_code=status.HTTP_200_OK, detail="No courses found")
        return feedbacks
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))

@app.post("/backend/createMCQ",status_code=status.HTTP_200_OK)
async def create_MCQ(req : Request):
    data = await req.json()
    mcq_id , questions,answers = create_MCQ_service(data)
    return {"MCQ_id": mcq_id, "questions":json.dumps(questions),"answers":answers}

@app.post("/backend/getMCQ",status_code=status.HTTP_200_OK)
async def get_MCQ(req: Request):
    data = await req.json()
    mcq_id = data['mcq_id']
    mcq = get_MCQ_service(mcq_id)
    if not mcq:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    return mcq

#GLIFIC APIs
@app.post("/backend/getCoursesGlific",status_code=status.HTTP_200_OK)
async def getCoursesGlific(req : Request):
    data = await req.json()
    user_id = data['user_id']
    company_id = get_company_by_user_service(user_id)
    if not company_id:
        print("here")
        raise HTTPException(status_code=404, detail="User not associated with any company")

    try:
        courses = get_all_courses_service(company_id=company_id[0])
        if not courses:
            raise HTTPException(status_code=404, detail="No courses found")
        course_titles = [course["title"] for course in courses]
        message = "Courses available to you are as follows:\n" + "\n".join(
            [f"{i+1}. {title}" for i, title in enumerate(course_titles)]
        )
        return {"message":message}
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))

@app.post("/backend/getLessonsGlific",status_code=status.HTTP_200_OK)
async def getLessonsGlific(req : Request):
    data = await req.json()
    course_number = int(data['course_number'])
    courses_name = data['courses_name']
    course_title = courses_name.split("\n")[course_number].split(".")[1].strip()
    course_id = get_course_id_by_name(course_title)
    lessons = get_lessons_service(course_id)
    if not lessons:
        raise HTTPException(status_code=status.HTTP_200_OK, detail="Lesson not found")
    lesson_titles = [lesson["title"] for lesson in lessons]
    message = "Lessons available to you are as follows:\n" + "\n".join(
        [f"{i+1}. {title}" for i, title in enumerate(lesson_titles)]
    )    
    return {"message": message}

@app.post("/backend/getAssessmentsGlific",status_code=status.HTTP_200_OK)
async def getAssessmentsGlific(req : Request):
    data = await req.json()
    lesson_number = int(data['lesson_number'])
    lessons_name = data['lessons_name']
    lesson_name = lessons_name.split("\n")[lesson_number].split(".")[1].strip()
    lesson_id = get_lesson_id_by_name(lesson_name)
    assessments = get_all_assessment_by_lesson_service(lesson_id)
    if not assessments:
        raise HTTPException(status_code=status.HTTP_200_OK, detail="Assessments not found")
    assessment_titles = [assessment['title'] for assessment in assessments]
    message = "Assessments available to you are as follows:\n" + "\n".join(
        [f"{i+1}. {title}" for i, title in enumerate(assessment_titles)]
    )    
    return {"message": message}

@app.post("/backend/getMCQGlific",status_code=status.HTTP_200_OK)
async def getMCQGlific(req : Request):
    data = await req.json()
    assessment_number = int(data['assessment_number'])
    assessments_name = data['assessments_name']
    assessments_name = assessments_name.split("\n")[assessment_number].split(".")[1].strip()
    assessment_id = get_assessment_id_by_name(assessments_name)
    MCQ_id = get_MCQ_by_assessment_service(assessment_id)
    if not MCQ_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCQ not found")
    res , correct_answers =  get_mcq_question_message(MCQ_id) 
    return {"message":res , "correct_answers":correct_answers}

@app.post("/backend/compareAnswers",status_code=status.HTTP_200_OK)
async def compareAnswers(req: Request):
    data = await req.json()
    user_answers = data["user_answers"]  # User's answers as a string
    correct_answers = data["correct_answers"]  # Correct answers as a string
    try:
        # Split the answers into lists
        user_answers_list = user_answers.split(",")
        correct_answers_list = correct_answers.split(",")
        # Check if the number of answers matches
        if len(user_answers_list) != len(correct_answers_list):
            return {"message": "Answer counts do not match. Please check your submission."}

        # Compare answers
        correct_count = sum(
            1 for u, c in zip(user_answers_list, correct_answers_list) if u == c
        )
        wrong_count = len(user_answers_list) - correct_count

        # Generate response message
        total_questions = len(correct_answers_list)
        message = f"You answered {correct_count} out of {total_questions} questions correctly. {wrong_count} answers were incorrect."

        return {"message": message,}

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}

@app.post("/backend/getFeedbackGlific",status_code=status.HTTP_200_OK)
async def getFeedbackGlific(req : Request):
    data = await req.json()
    course_number = int(data['course_number'])
    courses_name = data['courses_name']
    course_title = courses_name.split("\n")[course_number].split(".")[1].strip()
    course_id = get_course_id_by_name(course_title)
    feedback = get_feedback_questions_service(course_id)
    if not feedback:
        return {"message":"No feedback available for this course."}
    return  feedback
     
@app.post("/backend/addFeedback", status_code=status.HTTP_201_CREATED)
async def add_feedback(req: Request):
    try:
        data = await req.json()

        course_number = int(data['course_number'])
        courses_name = data['courses_name']
        course_title = courses_name.split("\n")[course_number].split(".")[1].strip()
        course_id = get_course_id_by_name(course_title)

        feedback_number = int(data["feedback_number"])
        feedback_questions = data['feedback_questions']
        feedback_question = feedback_questions.split("\n")[feedback_number].split(".")[1].strip()
        print("Feedback_Question : ", feedback_question)
        feedback_question_id = get_feedback_question_id_by_question(feedback_question)
        feedback_id = add_feedback_service(
            course_id, 
            feedback_question,
            feedback_question_id,
            user_id=data["user_id"], 
            feedback=data["feedback"]
        )

        return {"message": "Feedback added successfully", "feedback_id": feedback_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/backend/dashboard",status_code=status.HTTP_200_OK)
async def dashboard(req : Request):
    try:
        data = await req.json()
        company_id = data["company_id"]
        user_id = data["user_id"]
        result = get_dashboard_data_service(company_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/backend/initialize_progress",status_code=status.HTTP_200_OK)
async def initialize_progress(req : Request):
    """
    Initializes or resets progress for a user. If progress exists, reset it;
    otherwise, initialize progress for the first time.
    """
    data = await req.json()
    user_id = data.get("user_id")
    lesson_number = int(data['lesson_number'])
    lessons_name = data['lessons_name']
    lesson_name = lessons_name.split("\n")[lesson_number].split(".")[1].strip()
    lesson_id = get_lesson_id_by_name(lesson_name)
    # idx = get_lesson_PDF(lesson_id)
    idx = "vmartethics"
    try:
        # Check if the user already has progress in the user_qna_progress table
        result = get_question_count_voice(user_id, idx)        
        if result[0] > 0:
            # If progress exists, reset it
            reset_progress_qna_service(user_id, idx)
            return {"message": "Existing progress found. Progress reset successfully."}
        else:
            # If no progress exists, initialize it
            initialize_progress_qna_service(user_id,idx)
            return {"message": "No existing progress found. Progress initialized successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/backend/next_question",status_code=status.HTTP_200_OK)
async def get_next_question(req: Request):
    """Fetch the next unanswered question for a user."""
    data = await req.json()
    user_id = data.get("user_id")
    idx = "vmartethics"
    try:
        next_question = get_next_question_service(user_id,idx)
        return next_question
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/backend/submit_answer",status_code=status.HTTP_200_OK)
async def submit_answer(req:Request):
    """Submit a user's answer and mark the question as answered."""
    data = await req.json()
    user_response = data.get("user_response")
    user_id = data.get("user_id")
    qna_id = data.get("qna_id")
    try:
        submit_answer_service(user_response,user_id,qna_id)
        return {"message": "Answer submitted successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/backend/getLessonType", status_code=status.HTTP_200_OK)
async def get_lesson_content_type(req:Request):
    """
    API endpoint to retrieve the content type of a specific lesson.
    """
    data = await req.json()
    lesson_number = int(data['lesson_number'])
    lessons_name = data['lessons_name']
    lesson_name = lessons_name.split("\n")[lesson_number].split(".")[1].strip()
    lesson_id = get_lesson_id_by_name(lesson_name)
    try:
        content_type = get_lesson_content_type_service(lesson_id)
        return {"type":content_type}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@app.post("/backend/addSchool")
async def add_school(request: Request):
    school = await request.json()
    insert_query = """
        INSERT INTO schools (
            school_name, school_type, education_board, company_id, 
            principal_name, established_year, student_capacity, name, 
            phone, website, email, street_address, city, state, 
            description, zipcode, country
        ) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (
            school["school_name"],
            school["school_type"],
            school.get("education_board"),
            school["company_id"],
            school.get("principal_name"),
            school.get("established_year"),
            school["student_capacity"],
            school["name"],
            school["phone"],
            school["website"],
            school["email"],
            school["street_address"],
            school["city"],
            school["state"],
            school["description"],
            school["zipcode"],
            school["country"]
        ))
        connection.commit()
        return {"school_id": cursor.lastrowid}
    except Exception as e:
        connection.rollback()  # Rollback in case of error
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/viewSchool")
async def view_school(request: Request):
    try:
        school_id = request.query_params.get('school_id')
        if not school_id:
            raise HTTPException(status_code=400, detail="Missing school_id in query parameters")
        
        cursor.execute("SELECT * FROM schools WHERE school_id = %s", (school_id,))
        db_school = cursor.fetchone()
        
        if db_school is None:
            raise HTTPException(status_code=404, detail="School not found")
        
        school = {
            "school_id": db_school[0],
            "school_name": db_school[1],
            "school_type": db_school[2],
            "education_board": db_school[3],
            "company_id": db_school[4],
            "principal_name": db_school[5],
            "established_year": db_school[6],
            "student_capacity": db_school[7],
            "name": db_school[8],
            "phone": db_school[9],
            "website": db_school[10],
            "email": db_school[11],
            "street_address": db_school[12],
            "city": db_school[13],
            "state": db_school[14],
            "description": db_school[15],
            "zipcode": db_school[16],
            "country": db_school[17]
        }
        
        return school
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/backend/editSchool")
async def edit_school(request: Request):
    school_update = await request.json()
    school_id = school_update["school_id"]
    update_query = """
        UPDATE schools SET 
            school_name = %s,
            school_type = %s, 
            education_board = %s,
            company_id = %s, 
            principal_name = %s,
            established_year = %s,
            student_capacity = %s,
            name = %s, 
            phone = %s, 
            website = %s,
            email = %s, 
            street_address = %s, 
            city = %s, 
            state = %s, 
            description = %s, 
            zipcode = %s, 
            country = %s 
        WHERE school_id = %s
    """
    cursor.execute(update_query, (
        school_update["school_name"],
        school_update["school_type"], 
        school_update["education_board"],
        school_update["company_id"], 
        school_update["principal_name"],
        school_update["established_year"],
        school_update["student_capacity"],
        school_update["name"], 
        school_update["phone"], 
        school_update["website"],
        school_update["email"], 
        school_update["street_address"], 
        school_update["city"], 
        school_update["state"], 
        school_update["description"], 
        school_update["zipcode"], 
        school_update["country"],
        school_id
    ))
    connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="School not found")
    return {"message": "School updated successfully"}

@app.delete("/backend/deleteSchool")
async def delete_school(request: Request):
    school_id = request.query_params.get('school_id')
    if not school_id:
        raise HTTPException(status_code=400, detail="Missing school_id in query parameters")

    try:
       
        cursor.execute("""
            DELETE FROM school_assessments 
            WHERE course_id IN (
                SELECT course_id FROM school_courses WHERE school_id = %s
            )
        """, (school_id,))
        connection.commit()

        
        cursor.execute("""
            DELETE FROM school_presentations 
            WHERE lesson_id IN (
                SELECT lesson_id FROM school_lessons WHERE course_id IN (
                    SELECT course_id FROM school_courses WHERE school_id = %s
                )
            )
        """, (school_id,))
        connection.commit()

        
        cursor.execute("""
            DELETE FROM school_lessons 
            WHERE course_id IN (
                SELECT course_id FROM school_courses WHERE school_id = %s
            )
        """, (school_id,))
        connection.commit()

        
        cursor.execute("DELETE FROM school_courses WHERE school_id = %s", (school_id,))
        connection.commit()

        
        cursor.execute("DELETE FROM grades WHERE school_id = %s", (school_id,))
        connection.commit()

        
        cursor.execute("""
            DELETE FROM school_teachers 
            WHERE teacher_id IN (
                SELECT teacher_id FROM school_courses WHERE school_id = %s
            )
        """, (school_id,))
        connection.commit()

       
        delete_query = "DELETE FROM schools WHERE school_id = %s"
        cursor.execute(delete_query, (school_id,))
        connection.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="School not found")

        return {"message": "School and all related records deleted successfully"}
    except Exception as e:
        connection.rollback() 
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/backend/addGrades")
async def add_grades(request: Request):
    try:
        grade = await request.json()
        
        # Debugging: Print the incoming JSON data
        print(f"Received Grade Data: {grade}")
        
        # Ensure all necessary keys are present
        required_keys = ["school_id", "name", "capacity", "subjects", "description", "industry", "date_created"]
        for key in required_keys:
            if key not in grade:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        insert_query = """
            INSERT INTO grades (school_id, name, capacity, subjects, description, industry, date_created) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Debugging: Print the SQL query and parameters
        print(f"Executing query: {insert_query}")
        print(f"With parameters: ({grade['school_id']}, {grade['name']}, {grade['capacity']}, {grade['subjects']}, {grade['description']}, {grade['industry']}, {grade['date_created']})")
        
        cursor.execute(insert_query, (
            grade["school_id"], 
            grade["name"], 
            grade["capacity"], 
            grade["subjects"], 
            grade["description"], 
            grade["industry"],
            grade["date_created"]
        ))
        connection.commit()
        return {"grade_id": cursor.lastrowid}
    except HTTPException as e:
        return {"error": str(e)}
    except Exception as e:
        # Catch any other exceptions and return an error message
        return {"error": f"An unexpected error occurred: {str(e)}"}
    
@app.get("/backend/viewGrades")
async def view_grades(grade_id: int):
        cursor.execute("SELECT * FROM grades WHERE grade_id = %s", (grade_id,))
        db_grade = cursor.fetchone()
        if db_grade is None:
            raise HTTPException(status_code=404, detail="Grade not found")
        return db_grade

@app.put("/backend/editGrades")
async def edit_grades(request: Request):
    grade_update = await request.json()
    grade_id = grade_update["grade_id"]
    school_id = grade_update["school_id"]

    
    check_school_query = "SELECT 1 FROM schools WHERE school_id = %s"
    cursor.execute(check_school_query, (school_id,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=400, detail="Invalid school_id")

    
    update_query = """
        UPDATE grades SET 
            school_id = %s, 
            name = %s, 
            capacity = %s, 
            subjects = %s, 
            description = %s, 
            industry = %s 
        WHERE grade_id = %s
    """
    cursor.execute(update_query, (
        grade_update["school_id"], 
        grade_update["name"], 
        grade_update["capacity"], 
        grade_update["subjects"], 
        grade_update["description"], 
        grade_update["industry"], 
        grade_id
    ))
    connection.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Grade not found")

    return {"message": "Grade updated successfully"}

@app.delete("/backend/deleteGrades")
async def delete_grades(grade_id: int):
        delete_query = "DELETE FROM grades WHERE grade_id = %s"
        cursor.execute(delete_query, (grade_id,))
        connection.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Grade not found")
        return {"message": "Grade deleted successfully"}

@app.get("/backend/viewAllSchools/{company_id}", response_model=List[dict])
async def view_all_schools(company_id: int):
    try:
        cursor.execute("SELECT * FROM schools WHERE company_id = %s", (company_id,))
        schools = cursor.fetchall()
        
        if not schools:
            raise HTTPException(status_code=404, detail="No schools found for this company ID")
        
        return [
            {
                "school_id": school[0],
                "school_name": school[1],
                "school_type": school[2],
                "education_board": school[3],
                "company_id": school[4],
                "principal_name": school[5],
                "established_year": school[6],
                "student_capacity": school[7],
                "name": school[8],
                "phone": school[9],
                "website": school[10],
                "email": school[11],
                "street_address": school[12],
                "city": school[13],
                "state": school[14],
                "description": school[15],
                "zipcode": school[16],
                "country": school[17]
            } for school in schools
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/backend/school/addCourse")
async def add_course(request: Request):
    try:
        course = await request.json()
        
        # Check for required fields
        required_keys = ["name", "school_id", "grade_id", "description", "teacher_id", "capacity", "duration"]
        for key in required_keys:
            if key not in course:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        insert_query = """
            INSERT INTO school_courses (name, school_id, grade_id, description, teacher_id, capacity, duration) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            course["name"],
            course["school_id"],
            course["grade_id"],
            course["description"],
            course["teacher_id"],
            course["capacity"],
            course["duration"]
        ))
        connection.commit()
        return {"course_id": cursor.lastrowid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/school/viewCourse")
async def view_course(request: Request):
    try:
        course_id = request.query_params.get('course_id')
        if not course_id:
            raise HTTPException(status_code=400, detail="Missing course_id in query parameters")
        
        cursor.execute("SELECT * FROM school_courses WHERE course_id = %s", (course_id,))
        db_course = cursor.fetchone()
        
        if db_course is None:
            raise HTTPException(status_code=404, detail="Course not found")
        
        course = {
            "course_id": db_course[0],
            "name": db_course[1],
            "description": db_course[2],
            "teacher_id": db_course[3],
            "capacity": db_course[4],
            "duration": db_course[6]
        }
        
        return course
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/backend/school/editCourse")
async def edit_course(request: Request):
    try:
        course_update = await request.json()
        course_id = course_update["course_id"]

        # Check if course exists
        cursor.execute("SELECT 1 FROM school_courses WHERE course_id = %s", (course_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Course not found")

        update_query = """
            UPDATE school_courses SET 
                name = %s,
                description = %s,
                teacher_id = %s,
                capacity = %s,
                duration = %s
            WHERE course_id = %s
        """
        cursor.execute(update_query, (
            course_update["name"],
            course_update["description"],
            course_update["teacher_id"],
            course_update["capacity"],
            course_update["duration"],
            course_id
        ))
        connection.commit()
        return {"message": "Course updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/backend/school/deleteCourse")
async def delete_course(course_id: int):
    try:
        delete_query = "DELETE FROM school_courses WHERE course_id = %s"
        cursor.execute(delete_query, (course_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Course not found")
            
        return {"message": "Course deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/courses/{teacher_id}", response_model=List[dict])
async def get_teacher_courses(teacher_id: int):
    try:
        cursor.execute("SELECT * FROM school_courses WHERE teacher_id = %s", (teacher_id,))
        courses = cursor.fetchall()
        
        if not courses:
            raise HTTPException(status_code=404, detail="No courses found for this teacher")
        
        return [
            {
                "course_id": course[0],
                "name": course[1],
                "description": course[2],
                "teacher_id": course[3],
                "capacity": course[4],
                "duration": course[6]
            } for course in courses
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/courses/{school_id}", response_model=List[dict])
async def get_teacher_courses(school_id: int):
    try:
        cursor.execute("SELECT * FROM school_courses WHERE school_id = %s", (school_id,))
        courses = cursor.fetchall()
        
        if not courses:
            raise HTTPException(status_code=404, detail="No courses found for this teacher")
        
        return [
            {
                "course_id": course[0],
                "name": course[1],
                "description": course[2],
                "teacher_id": course[3],
                "capacity": course[4],
                "duration": course[6]
            } for course in courses
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/backend/school/addTeacher")
async def add_teacher(request: Request):
    try:
        teacher = await request.json()
        
        # Check for required fields
        required_keys = ["name", "email", "subject", "qualification", "experience", "department"]
        for key in required_keys:
            if key not in teacher:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        insert_query = """
            INSERT INTO school_teachers (name, email, subject, qualification, experience, department) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            teacher["name"],
            teacher["email"],
            teacher["subject"],
            teacher["qualification"],
            teacher["experience"],
            teacher["department"]
        ))
        connection.commit()
        return {"id": cursor.lastrowid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/school/viewTeacher")
async def view_teacher(request: Request):
    try:
        teacher_id = request.query_params.get('teacher_id')
        if not teacher_id:
            raise HTTPException(status_code=400, detail="Missing teacher_id in query parameters")
        
        cursor.execute("SELECT * FROM school_teachers WHERE teacher_id = %s", (teacher_id,))
        db_teacher = cursor.fetchone()
        
        if db_teacher is None:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        teacher = {
            "teacher_id": db_teacher[0],
            "name": db_teacher[1],
            "email": db_teacher[2],
            "subject": db_teacher[3],
            "qualification": db_teacher[4],
            "experience": db_teacher[5],
            "department": db_teacher[6]
        }
        
        return teacher
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/backend/school/editTeacher")
async def edit_teacher(request: Request):
    try:
        teacher_update = await request.json()
        teacher_id = teacher_update["teacher_id"]

        # Check if teacher exists
        cursor.execute("SELECT 1 FROM school_teachers WHERE teacher_id = %s", (teacher_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Teacher not found")

        update_query = """
            UPDATE school_teachers SET 
                name = %s,
                email = %s,
                subject = %s,
                qualification = %s,
                experience = %s,
                department = %s
            WHERE teacher_id = %s
        """
        cursor.execute(update_query, (
            teacher_update["name"],
            teacher_update["email"],
            teacher_update["subject"],
            teacher_update["qualification"],
            teacher_update["experience"],
            teacher_update["department"],
            teacher_id
        ))
        connection.commit()
        return {"message": "Teacher updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/backend/school/deleteTeacher")
async def delete_teacher(teacher_id: int):
    try:

        cursor.execute("SELECT 1 FROM school_courses WHERE teacher_id = %s", (teacher_id,))
        if cursor.fetchone() is not None:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete teacher with associated courses. Please reassign or delete the courses first."
            )

        delete_query = "DELETE FROM school_teachers WHERE teacher_id = %s"
        cursor.execute(delete_query, (teacher_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Teacher not found")
            
        return {"message": "Teacher deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teachers/department/{department}", response_model=List[dict])
async def get_department_teachers(department: str):
    try:
        cursor.execute("SELECT * FROM school_teachers WHERE department = %s", (department,))
        teachers = cursor.fetchall()
        
        if not teachers:
            raise HTTPException(status_code=404, detail="No teachers found in this department")
        
        return [
            {
                "teacher_id": teacher[0],
                "name": teacher[1],
                "email": teacher[2],
                "subject": teacher[3],
                "qualification": teacher[4],
                "experience": teacher[5],
                "department": teacher[6]
            } for teacher in teachers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/teachers", response_model=List[dict])
async def get_all_teachers():
    try:
        cursor.execute("SELECT * FROM school_teachers")
        teachers = cursor.fetchall()
        
        return [
            {
                "teacher_id": teacher[0],
                "name": teacher[1],
                "email": teacher[2],
                "subject": teacher[3],
                "qualification": teacher[4],
                "experience": teacher[5],
                "department": teacher[6]
            } for teacher in teachers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/backend/school/addLesson")
async def add_lesson(request: Request):
    try:
        lesson = await request.json()
        
        required_keys = ["course_id", "title", "content", "duration", "resources"]
        for key in required_keys:
            if key not in lesson:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        if not lesson["title"].strip() or len(lesson["title"]) > 255:
            raise HTTPException(
                status_code=400,
                detail="Title cannot be empty and must be less than 255 characters"
            )
        
        duration = str(lesson["duration"]).lower().replace(' ', '')
        if not (duration.endswith('hrs') or duration.endswith('hr')):
            raise HTTPException(
                status_code=400,
                detail="Duration must be specified in hours (e.g., '6 hrs')"
            )
        
        try:
            hours = float(duration.rstrip('hrs').rstrip('hr'))
            if hours <= 0 or hours > 24:
                raise HTTPException(
                    status_code=400,
                    detail="Duration must be between 0 and 24 hours"
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid duration format"
            )
        
        if not isinstance(lesson["resources"], dict):
            raise HTTPException(
                status_code=400,
                detail="Resources must be a JSON object"
            )
        
        cursor.execute(
            "SELECT 1 FROM school_courses WHERE course_id = %s",
            (lesson["course_id"],)
        )
        if cursor.fetchone() is None:
            raise HTTPException(
                status_code=404,
                detail=f"Course with ID {lesson['course_id']} not found"
            )
        cursor.execute(
            "SELECT 1 FROM school_lessons WHERE course_id = %s AND title = %s",
            (lesson["course_id"], lesson["title"])
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=400,
                detail="A lesson with this title already exists in this course"
            )
        
        # Insert the lesson into the database
        insert_query = """
            INSERT INTO school_lessons 
            (course_id, title, content, duration, resources, status, created_at, updated_at) 
            VALUES (%s, %s, %s, %s, %s, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        
        # Convert resources to JSON string
        resources_json = json.dumps(lesson["resources"])
        
        cursor.execute(insert_query, (
            lesson["course_id"],
            lesson["title"].strip(),
            lesson["content"],
            lesson["duration"],
            resources_json,
        ))
        connection.commit()
        
        lesson_id = cursor.lastrowid
        
        return {
            "lesson_id": lesson_id,
            "message": "Lesson created successfully"
        }
        
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in resources field"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while creating the lesson: {str(e)}"
        )

@app.get("/backend/school/viewLesson")
async def view_lesson(lesson_id: int):
    try:
        cursor.execute("SELECT * FROM school_lessons WHERE lesson_id = %s", (lesson_id,))
        db_lesson = cursor.fetchone()
        
        if db_lesson is None:
            raise HTTPException(status_code=404, detail="Lesson not found")
        
        lesson = {
            "lesson_id": db_lesson[0],
            "course_id": db_lesson[1],
            "title": db_lesson[2],
            "content": db_lesson[3],
            "duration": db_lesson[4],
            "resources": db_lesson[5],
            "status": db_lesson[6]
        }
        return lesson
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/backend/school/editLesson")
async def edit_lesson(request: Request):
    try:
        lesson_update = await request.json()
        lesson_id = lesson_update["lesson_id"]

        cursor.execute("SELECT 1 FROM school_lessons WHERE lesson_id = %s", (lesson_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Lesson not found")

        update_query = """
            UPDATE school_lessons SET 
                course_id = %s,
                title = %s,
                content = %s,
                duration = %s,
                resources = %s,
                status = %s
            WHERE lesson_id = %s
        """

        # Convert resources to JSON string
        resources_json = json.dumps(lesson_update["resources"])

        cursor.execute(update_query, (
            lesson_update["course_id"],
            lesson_update["title"],
            lesson_update["content"],
            lesson_update["duration"],
            resources_json,
            lesson_update["status"],
            lesson_id
        ))
        connection.commit()
        return {"message": "Lesson updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/backend/school/deleteLesson")
async def delete_lesson(lesson_id: int):
    try:
        cursor.execute("SELECT 1 FROM school_presentations WHERE lesson_id = %s", (lesson_id,))
        if cursor.fetchone() is not None:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete lesson with associated presentations. Please delete the presentations first."
            )

        delete_query = "DELETE FROM school_lessons WHERE lesson_id = %s"
        cursor.execute(delete_query, (lesson_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Lesson not found")
            
        return {"message": "Lesson deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backend/school/addPresentation")
async def add_presentation(request: Request):
    try:
        presentation = await request.json()
        
        required_keys = ["lesson_id", "title", "content", "media_urls", "duration"]
        for key in required_keys:
            if key not in presentation:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        cursor.execute("SELECT 1 FROM school_lessons WHERE lesson_id = %s", (presentation["lesson_id"],))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=400, detail="Invalid lesson_id")
        
        # Ensure media_urls is a valid JSON object
        try:
            media_urls_json = json.dumps(presentation["media_urls"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in media_urls field")
        
        insert_query = """
            INSERT INTO school_presentations (lesson_id, title, content, media_urls, duration, status) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            presentation["lesson_id"],
            presentation["title"],
            presentation["content"],
            media_urls_json,
            presentation["duration"],
            "active"  # Default status
        ))
        connection.commit()
        return {"presentation_id": cursor.lastrowid}
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in media_urls field"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/school/viewPresentation")
async def view_presentation(presentation_id: int):
    try:
        cursor.execute("SELECT * FROM school_presentations WHERE presentation_id = %s", (presentation_id,))
        db_presentation = cursor.fetchone()
        
        if db_presentation is None:
            raise HTTPException(status_code=404, detail="Presentation not found")
        
        presentation = {
            "presentation_id": db_presentation[0],
            "lesson_id": db_presentation[1],
            "title": db_presentation[2],
            "content": db_presentation[3],
            "media_urls": db_presentation[4],
            "duration": db_presentation[5],
            "status": db_presentation[6]
        }
        return presentation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backend/school/addPresentation")
async def add_presentation(request: Request):
    try:
        presentation = await request.json()
        
        required_keys = ["lesson_id", "title", "content", "media_urls", "duration"]
        for key in required_keys:
            if key not in presentation:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        cursor.execute("SELECT 1 FROM school_lessons WHERE lesson_id = %s", (presentation["lesson_id"],))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=400, detail="Invalid lesson_id")
        
        # Ensure media_urls is a valid JSON object
        try:
            media_urls_json = json.dumps(presentation["media_urls"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in media_urls field")
        
        insert_query = """
            INSERT INTO school_presentations (lesson_id, title, content, media_urls, duration, status) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            presentation["lesson_id"],
            presentation["title"],
            presentation["content"],
            media_urls_json,
            presentation["duration"],
            "active"  # Default status
        ))
        connection.commit()
        return {"presentation_id": cursor.lastrowid}
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in media_urls field"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/backend/school/editPresentation")
async def edit_presentation(request: Request):
    try:
        presentation_update = await request.json()
        presentation_id = presentation_update["presentation_id"]

        cursor.execute("SELECT 1 FROM school_presentations WHERE presentation_id = %s", (presentation_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Presentation not found")

        cursor.execute("SELECT 1 FROM school_lessons WHERE lesson_id = %s", (presentation_update["lesson_id"],))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=400, detail="Invalid lesson_id")

        # Ensure media_urls is a valid JSON object
        try:
            media_urls_json = json.dumps(presentation_update["media_urls"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in media_urls field")

        update_query = """
            UPDATE school_presentations SET 
                lesson_id = %s,
                title = %s,
                content = %s,
                media_urls = %s,
                duration = %s,
                status = %s
            WHERE presentation_id = %s
        """
        cursor.execute(update_query, (
            presentation_update["lesson_id"],
            presentation_update["title"],
            presentation_update["content"],
            media_urls_json,
            presentation_update["duration"],
            presentation_update["status"],
            presentation_id
        ))
        connection.commit()
        return {"message": "Presentation updated successfully"}
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in media_urls field"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/backend/school/deletePresentation")
async def delete_presentation(presentation_id: int):
    try:
        delete_query = "DELETE FROM school_presentations WHERE presentation_id = %s"
        cursor.execute(delete_query, (presentation_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Presentation not found")
            
        return {"message": "Presentation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backend/school/addAssessment")
async def add_assessment(request: Request):
    try:
        assessment = await request.json()
        
        # Check for required fields
        required_keys = ["course_id", "title", "type", "questions", "total_marks", "duration"]
        for key in required_keys:
            if key not in assessment:
                raise HTTPException(status_code=400, detail=f"Missing required field: {key}")
        
        cursor.execute("SELECT 1 FROM school_courses WHERE course_id = %s", (assessment["course_id"],))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=400, detail="Invalid course_id")
        
        # Ensure questions is a valid JSON object
        try:
            questions_json = json.dumps(assessment["questions"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in questions field")
        
        insert_query = """
            INSERT INTO school_assessments (course_id, title, type, questions, total_marks, duration, status) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            assessment["course_id"],
            assessment["title"],
            assessment["type"],
            questions_json,
            assessment["total_marks"],
            assessment["duration"],
            "active"  # Default status
        ))
        connection.commit()
        return {"assessment_id": cursor.lastrowid}
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in questions field"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/school/viewAssessment")
async def view_assessment(assessment_id: int):
    try:
        cursor.execute("SELECT * FROM school_assessments WHERE assessment_id = %s", (assessment_id,))
        db_assessment = cursor.fetchone()
        
        if db_assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = {
            "assessment_id": db_assessment[0],
            "course_id": db_assessment[1],
            "title": db_assessment[2],
            "type": db_assessment[3],
            "questions": db_assessment[4],
            "total_marks": db_assessment[5],
            "status": db_assessment[6]
        }
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/backend/school/editAssessment")
async def edit_assessment(request: Request):
    try:
        assessment_update = await request.json()
        assessment_id = assessment_update["assessment_id"]

        # Check if assessment exists
        cursor.execute("SELECT 1 FROM school_assessments WHERE assessment_id = %s", (assessment_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Assessment not found")

        # Ensure questions is a valid JSON object
        try:
            questions_json = json.dumps(assessment_update["questions"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format in questions field")

        update_query = """
            UPDATE school_assessments SET 
                course_id = %s,
                title = %s,
                type = %s,
                questions = %s,
                total_marks = %s,
                duration = %s,
                status = %s
            WHERE assessment_id = %s
        """
        cursor.execute(update_query, (
            assessment_update["course_id"],
            assessment_update["title"],
            assessment_update["type"],
            questions_json,
            assessment_update["total_marks"],
            assessment_update["duration"],
            assessment_update["status"],
            assessment_id
        ))
        connection.commit()
        return {"message": "Assessment updated successfully"}
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in questions field"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/backend/school/deleteAssessment")
async def delete_assessment(assessment_id: int):
    try:
        delete_query = "DELETE FROM school_assessments WHERE assessment_id = %s"
        cursor.execute(delete_query, (assessment_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assessment not found")
            
        return {"message": "Assessment deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/backend/dashboard/{company_id}")
async def get_dashboard():
    try:

        cursor.execute("SELECT COUNT(*) FROM schools")
        total_schools = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM school_teachers")
        total_teachers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM school_courses")
        total_courses = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(student_capacity) FROM schools")
        total_students = cursor.fetchone()[0] or 0

        cursor.execute("SELECT city, COUNT(*) FROM schools GROUP BY city")
        region_wise_users = [
            {"city": city, "count": count} for city, count in cursor.fetchall()
        ]

        #cursor.execute("SELECT AVG(performance) FROM school_teachers")
        #teacher_completion_rate = round(cursor.fetchone()[0] or 0)


       # cursor.execute("SELECT name, performance FROM school_teachers ORDER BY performance DESC LIMIT 4")
        #top_teachers = [
       #    {"name": name, "performance": performance} for name, performance in cursor.fetchall()]
        

        return {
            "total_schools": total_schools,
            "total_teachers": total_teachers,
            "total_courses": total_courses,
            "total_students": total_students,
            "region_wise_users": region_wise_users
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)