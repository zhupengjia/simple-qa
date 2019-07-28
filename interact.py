#!/usr/bin/env python
from xlnet_qa.qa_server import QAServer
from pytorch_transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
s = QAServer("data/wiki/wiki.bz2")


