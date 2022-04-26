# FROM python:3.7-bullseye

FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get -y update && apt-get install -y software-properties-common
RUN apt-add-repository non-free
RUN apt-get -y update && apt-get -y install ffmpeg unrar

RUN pip install tqdm gym gym[atari] gym[accept-rom-license] pygame ale-py torch torchvision

COPY dqn++.py ./

CMD python dqn++.py