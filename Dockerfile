FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

WORKDIR /code

COPY requirements.txt .

# Install core dependencies.
RUN apt-get update && apt-get install -y libpq-dev build-essential

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main.py"]
