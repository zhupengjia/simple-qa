FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN conda install nltk flask &&\
    conda install -c conda-forge python-xapian

# python packages
RUN python3 -c "import nltk;nltk.download('wordnet')" &&\
    pip3 install --no-cache transformers

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD simple_qa simple_qa

ADD interact.py interact.py

CMD python interact.py ${BOT_INPUT} --model ${BOT_MODEL} --scorelimit ${BOT_SCORELIMIT}  --backend restful --port 5000
#CMD tail -f /dev/null
