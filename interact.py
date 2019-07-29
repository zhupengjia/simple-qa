#!/usr/bin/env python
import argparse, sys

parser = argparse.ArgumentParser(description='Read comprehension using XLNet pretrained model')
parser.add_argument('-i', '--in', dest='input', help="input .txt, .gz, .bz2 file")
parser.add_argument('-r', '--recreate', dest="recreate", action='store_true', help="set to recreate index")
parser.add_argument('-p', '--port', dest='port', help="listen port")
args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    parser.exit()

from xlnet_qa.qa_server import QAServer

s = QAServer(args.input, recreate=args.recreate, port=args.port)

print(s.search("quiver"))
