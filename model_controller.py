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

def train_on_part_data(model, n):
    m = Model(n)
    m.copy_partition_data(model)
    m.train()
    return m.eval(), m

args  = sys.argv 
nargs = len(args)

file_agg = []

for arg in args:
    if arg[-4:] == ".txt":
        file_txt = open(arg, "r").read()
        file_agg += file_txt.split(' ')

best_perp, best_model = train_on_method_tokens(file_agg, 3)
for n in range(5, 9, 2):
    perplexity, model = train_on_part_data(best_model, n)
    if perplexity < best_perp:
        best_model = model
        best_perp  = perplexity

print("Perplexity of model: ", best_perp, " and n of model: ", best_model.n)
sample = best_model.get_sample()
print("Ground Truth: ", sample)
print("Prediction: ", ' '.join(best_model.predict(sample[:best_model.n-1], 100)))
