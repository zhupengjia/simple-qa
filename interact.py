#!/usr/bin/env python
import argparse, sys

parser = argparse.ArgumentParser(description='Read comprehension using XLNet pretrained model')
parser.add_argument('-d', '--device', dest="device", default="cuda:0", help="choose to use cpu or cuda:x, default is cuda:0")
parser.add_argument('-r', '--recreate', dest="recreate", action='store_true', help="set to recreate index")
parser.add_argument('-b', '--before', dest="before", default=0, help="prefiltered texts include N lines before related text, default is 0")
parser.add_argument('-a', '--after', dest="after", default=0, help="prefiltered texts include N lines after related text, default is 0")
parser.add_argument('-p', '--port', dest='port', default=5002, help="listen port, default is 5002")
parser.add_argument('-m', '--model', dest="model", required=True, help="path of model, must be a directory and contains file 'pytorch_model.bin' and 'config.json'")
parser.add_argument('input', help=".txt, .gz, .bz2 file path")
args = parser.parse_args()

from xlnet_qa.qa_server import QAServer

s = QAServer(file_path=args.input,
             model_path=args.model,
             device=args.device,
             recreate=args.recreate,
             port=args.port)

s.answer("When were the Normans in Normandy?")
