FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY libs/ ./libs/
COPY weights/ ./weights/

# Entry point
EXPOSE 8501
CMD ["python3", "-m", "streamlit", "run", "src/ai_image_dashboard/app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
