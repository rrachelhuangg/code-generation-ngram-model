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
for n in range(1, 8):
    perplexity, model = train_on_method_tokens(file_agg, n)
    if perplexity < best_perp:
        best_model = model
        best_perp = perplexity

print("Perplexity of model: ", best_perp, " and n of model: ", best_model.n)
print("Random Prediction: ", ' '.join(best_model.predict(best_model.get_sample(), 100)))
