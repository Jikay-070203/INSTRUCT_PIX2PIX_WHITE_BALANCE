FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt pip và một số công cụ cơ bản
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements nếu có
COPY requirements.txt .

# Cài các dependency Python
RUN pip install --upgrade pip \
    && pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Thiết lập biến môi trường cho CUDA
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Port cho FastAPI
EXPOSE 8000

# server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

