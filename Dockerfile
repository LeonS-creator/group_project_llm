# ---- Dockerfile for GrantMatch Lite (Gradio) ----
FROM python:3.11-slim

# System deps for lxml/readability-lxml
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libxml2     libxslt1.1     libxml2-dev     libxslt1-dev     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app into the image
COPY grant_chatbot.py /app/grant_chatbot.py

# Expose Gradio default port
EXPOSE 7860

# Gradio needs to listen on 0.0.0.0 in Docker
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Launch the app
CMD ["python", "grant_chatbot.py"]
