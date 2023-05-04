FROM python:3.8

COPY requirements.txt .

# Install core dependencies
RUN apt-get update && apt-get install -y libpq-dev build-essential

# Make sure we're using the latest pip version
RUN python3 -m pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY main.py .
CMD ["python3", "./main.py"]
