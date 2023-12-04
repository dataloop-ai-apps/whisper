FROM dataloopai/dtlpy-agent:gpu.cuda.11.5.py3.8.pytorch2
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

USER 1000

COPY ./ /tmp/app
WORKDIR /tmp/app


RUN pip3 install --user -r requirements.txt

# docker build -t dataloopai/whisper-gpu.cuda.11.5.py3.8.pytorch2:1.0.1 .
# docker push dataloopai/whisper-gpu.cuda.11.5.py3.8.pytorch2:1.0.1