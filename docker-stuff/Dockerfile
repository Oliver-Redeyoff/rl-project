FROM python:3.6-bullseye

RUN apt-get -y update && apt-get install -y software-properties-common
RUN apt-add-repository non-free
RUN apt-get -y update && apt-get -y install ffmpeg unrar

ENV CODE_DIR /root/code

COPY . $CODE_DIR/baselines
WORKDIR $CODE_DIR/baselines

RUN pip install atari-py

RUN wget "http://www.atarimania.com/roms/Roms.rar" && \
    mkdir roms && \
    cd roms && \
    unrar x ../Roms.rar

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install 'tensorflow < 2' && \
    pip install -e .[test] && \
    python -m atari_py.import_roms roms


CMD "/bin/bash"