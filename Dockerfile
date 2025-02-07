# ใช้ Base Image ของ Python ที่เล็กและเร็วกว่า
FROM python:3.9-slim

# เปิดพอร์ต 8501 สำหรับ Streamlit
EXPOSE 8501

# กำหนด working directory ใน container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt และติดตั้ง dependencies
COPY requirements.txt ./

# ติดตั้ง pip และใช้ --extra-index-url เพื่อโหลด PyTorch + ไลบรารีอื่น ๆ
RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# คัดลอกไฟล์ทั้งหมดจากโปรเจกต์ไปยัง container
COPY . .

# รัน Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
