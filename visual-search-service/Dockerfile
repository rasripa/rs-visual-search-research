FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY global_image_id_to_record.json .
COPY global_text_id_to_record.json .
COPY image.index .
COPY text.index .
COPY services.py .
COPY streamlit.py .
EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=80", "--server.address=0.0.0.0"]