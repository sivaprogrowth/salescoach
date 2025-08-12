FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (add portaudio19-dev for PyAudio)
RUN apt-get update && apt-get install -y ffmpeg gcc portaudio19-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Install supervisor
RUN pip install supervisor

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Add supervisor config
COPY supervisord.conf /etc/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisord.conf"]