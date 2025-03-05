FROM python:3.11.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app.py .

EXPOSE 8501

ENV OPENAI_API_KEY=$OPENAI_API_KEY

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
