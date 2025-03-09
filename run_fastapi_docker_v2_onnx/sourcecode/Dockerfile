# Chọn image base
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các tệp vào container
COPY . /app

# Cài đặt các thư viện yêu cầu
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt lại OpenCV (nếu cần)
RUN pip uninstall -y opencv-python-headless && pip install opencv-python

# Cài đặt bổ sung các package bị thiếu
RUN pip install accelerate python-multipart

# Cập nhật hệ thống và cài đặt các thư viện cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt ONNX Runtime (hoặc các thư viện khác nếu cần)
RUN pip install onnxruntime pillow numpy

# Cổng expose
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
