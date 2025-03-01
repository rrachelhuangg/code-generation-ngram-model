##  
# AUTHOR   : Rachel Huang, Jackson Taylor
# CREATED  : 20-2-2025
# EDITED   : 1-3-2025
# CONTAINS : Controller code to run model according to project specifications
##

import os
import sys
import json
from model import Model

def train_on_method_tokens(tokens, n):
    m = Model(n)
    m.partition_data(file_agg)
    m.train()
    return m.eval(m.validation_data), m

def train_on_part_data(model, n):
    m = Model(n)
    m.copy_partition_data(model)
    m.train()
    return m.eval(m.validation_data), m

args  = sys.argv 
nargs = len(args)

file_agg = []

for arg in args:
    if arg[-4:] == ".txt":
        file_txt = open(arg, "r").read()
        file_agg += file_txt.split(' ')

model_performances = {}
model_performances["n"] = {}
best_perp, best_model = train_on_method_tokens(file_agg, 3)
model_performances["n"][3] = best_perp
for n in range(5, 10, 2):
    perplexity, model = train_on_part_data(best_model, n)
    model_performances["n"][n] = perplexity
    if perplexity < best_perp:
        best_model = model
        best_perp  = perplexity

test_perp  = best_model.eval(best_model.test_data)
sample     = best_model.get_sample(best_model.test_data)
gt         = ' '.join(sample)
context    = sample[:best_model.n-1]
prediction = ' '.join(best_model.predict(sample[:best_model.n-1], 100))

model_performances["best"]                    = {}
model_performances["best"]["n"]               = best_model.n
model_performances["best"]["Test Perplexity"] = test_perp
model_performances["best"]["Ground Truth"]    = gt
model_performances["best"]["Context Window"]  = context
model_performances["best"]["Prediction"]      = prediction

with open('results_model.json', 'w') as outfile:
    json.dump(model_performances, outfile)
