FROM python:3.11.10
 
WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    libjack-dev \
    gcc \
    python3-dev \
&& rm -rf /var/lib/apt/lists/*
 
ENV PYAUDIO_HOME=/usr
ENV SDL_AUDIODRIVER=dummy
 
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
COPY app.py .
 
EXPOSE 8501
 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
