import os
import sys
from model import Model

def train_on_method_tokens(tokens):
    pass

args  = sys.argv
nargs = len(args)

file_agg = []

for arg in args:
    if arg[-4:] == ".txt":
        file_txt = open(arg, "r").read()
        file_agg += file_txt.split(' ')

m = Model(3)
m.partition_data(file_agg)
m.train()
print("Perplexity of model: ", m.eval())
print("Predicted sentence given [public, static]: ", m.predict(["public", "static"]))