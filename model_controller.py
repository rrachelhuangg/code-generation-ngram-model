##  
# AUTHOR   : Rachel Huang, Jackson Taylor
# CREATED  : 20-2-2025
# EDITED   : 23-2-2025
# CONTAINS : Controller code to run model according to project specifications
##

import os
import sys
from model import Model

def train_on_method_tokens(tokens, n):
    m = Model(n)
    m.partition_data(file_agg)
    m.train()
    return m.eval(), m

args  = sys.argv 
nargs = len(args)

file_agg = []

for arg in args:
    if arg[-4:] == ".txt":
        file_txt = open(arg, "r").read()
        file_agg += file_txt.split(' ')

best_model = None
best_perp  = 100000
for n in range(2, 8):
    perplexity, model = train_on_method_tokens(file_agg, n)
    if perplexity < best_perp:
        best_model = model
        best_perp = perplexity

print("Perplexity of model: ", best_perp, " and n of model: ", best_model.n)
# sample = best_model.get_sample()
# print("GROUND TRUTH: ", sample)
# print("PREDICT: ", ' '.join(best_model.predict(sample, 100)))
print("Random Prediction: ", ' '.join(best_model.predict(best_model.get_sample(), 100)))
