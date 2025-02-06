FROM python:latest


RUN apt-get update
RUN pip install --no-cache-dir -U pip

WORKDIR /project
COPY . /project/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/project/core/run.py"]