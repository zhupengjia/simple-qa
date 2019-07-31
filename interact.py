#!/usr/bin/env python
import argparse, sys

parser = argparse.ArgumentParser(description='Read comprehension using XLNet pretrained model')
parser.add_argument('--device', dest="device", default="cuda:0", help="choose to use cpu or cuda:x, default is cuda:0")
parser.add_argument('--recreate', dest="recreate", action='store_true', help="set to recreate index")
parser.add_argument('--before', dest="before", default=0, help="prefiltered texts include N lines before related text, default is 0")
parser.add_argument('--after', dest="after", default=0, help="prefiltered texts include N lines after related text, default is 0")
parser.add_argument('-p', '--port', dest='port', default=5002, help="listen port, default is 5002")
parser.add_argument('--backend', dest='backend', default='shell', help="choose for backend from: shell, restful, default is shell")
parser.add_argument('-m', '--model', dest="model", required=True, help="path of model, must be a directory and contains file 'pytorch_model.bin' and 'config.json'")
parser.add_argument('input', help=".txt, .gz, .bz2 file path")
args = parser.parse_args()

from xlnet_qa.backend import Backend

s = Backend(backend_type=args.backend,
             file_path=args.input,
             model_path=args.model,
             device=args.device,
             recreate=args.recreate,
             port=args.port)
s.run()
