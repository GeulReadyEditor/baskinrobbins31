# https://hub.docker.com/r/theeluwin/ubuntu-konlpy/dockerfile
FROM ubuntu:latest

# apt init
ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata g++ git curl

# installing java jdk and java jre
RUN apt-get install -y openjdk-8-jdk

# installing python3 and pip3
RUN apt-get install -y python3-pip python3-dev

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip3 install --upgrade pip

# apt cleanse
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# timezone
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# install python packages
RUN pip install konlpy
RUN cd /app && \
    curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./

CMD ["python", "start_flask.py"]