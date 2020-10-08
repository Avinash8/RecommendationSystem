FROM python:3.7-slim

RUN mkdir /opt/hello_app

WORKDIR /opt/hello_app

ADD requirements.txt .

RUN pip install -r requirements.txt && pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

ADD . .