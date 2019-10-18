FROM pytorch/pytorch:latest
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends python3-numpy python3-scipy python3-wheel python3-pip python3-tqdm python3-yaml python3-nltk python3-flask python3-setuptools python3-regex python3-requests python3-boto3 cython3 python3-xapian git&&\
    ln -s /usr/bin/python3 /usr/local/bin/python &&\
    ln -s /usr/bin/pip3 /usr/local/bin/pip &&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN python3 -c "import nltk;nltk.download('wordnet')" &&\
    pip3 install --no-cache torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html &&\
    pip3 install --no-cache transformers

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD simple_qa simple_qa

ADD interact.py interact.py

CMD python interact.py ${BOT_INPUT} --model ${BOT_MODEL} --scorelimit ${BOT_SCORELIMIT}  --backend restful --port 5000
#CMD tail -f /dev/null
