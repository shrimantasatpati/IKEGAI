FROM python:3.11

RUN mkdir app
WORKDIR /app

ADD /app /app

RUN pip install -r requirements.txt
EXPOSE 6069
