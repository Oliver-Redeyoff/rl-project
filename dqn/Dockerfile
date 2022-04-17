FROM python:3.7-bullseye

RUN apt-get -y update && apt-get install -y software-properties-common
RUN apt-add-repository non-free
RUN apt-get -y update && apt-get -y install ffmpeg unrar

COPY dqn.py ./

RUN pip install tqdm gym gym[atari] gym[accept-rom-license] pygame ale-py torch

CMD "/bin/bash"