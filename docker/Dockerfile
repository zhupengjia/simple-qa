FROM debian:buster-slim
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends python3-numpy python3-joblib python3-tqdm python3-nltk python3-flask python3-pip python3-xapian python3-setuptools python3-regex python3-requests python3-boto3 cython3 &&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN python3 -c "import nltk;nltk.download('wordnet')" &&\
    pip3 install --no-cache transformers

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD simple_qa simple_qa

ADD interact.py interact.py

CMD python interact.py ${BOT_INPUT} --model ${BOT_MODEL} --scorelimit ${BOT_SCORELIMIT}  --backend ${BOT_BACKEND} --port 5000
#CMD tail -f /dev/null
