FROM python:3.10

WORKDIR /app

COPY ./app .

COPY ./requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install -r requirements.txt --default-timeout=1000 --no-cache-dir

RUN python setup_embeddings.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]