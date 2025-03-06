# Dùng image Python cơ bản
FROM python:3.11.7

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cập nhật hệ thống và cài đặt các thư viện cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements và cài đặt thư viện trước
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt bổ sung các package bị thiếu
RUN pip install accelerate python-multipart

# Copy mã nguồn FastAPI vào container
COPY apptrism.py /app/apptrism.py
COPY process_image.py /app/process_image.py
COPY requirements.txt /app/requirements.txt

# Copy model vào container
COPY model_repository/instruct-pix2pix /app/model

# Cài đặt lại OpenCV (nếu cần)
RUN pip uninstall -y opencv-python-headless && pip install opencv-python

# Chạy FastAPI
CMD ["uvicorn", "apptrism:app", "--host", "0.0.0.0", "--port", "8000"]


