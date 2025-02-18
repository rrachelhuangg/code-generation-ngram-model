import os
import sys

def train_on_method_tokens(tokens):
    pass

args  = sys.argv
nargs = len(args)

for arg in args:
    if arg[-3:] == ".txt":
        file_txt = open(arg, "r").read()
        methods = file_txt.split("\n")
