FROM python:3.8

WORKDIR /code

COPY src/ .

CMD ["python", "./main.py"]
