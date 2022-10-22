FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR ~/build

RUN chmod 1777 /tmp
RUN pip install --upgrade pip cmake
#RUN apt update && apt install -y build-essential
RUN apt install libgl1
COPY ./context/requirements.txt .
RUN pip install -r requirements.txt

RUN rm -rf ~/build

COPY . /project
WORKDIR /project
