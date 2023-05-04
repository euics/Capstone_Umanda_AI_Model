FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

# Install core dependencies.
RUN apt-get update && apt-get install -y libpq-dev build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
ENTRYPOINT ["python3"]
CMD ["app.py"]
