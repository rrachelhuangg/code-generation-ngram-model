##  
# AUTHOR   : Rachel Huang, Jackson Taylor
# CREATED  : 20-2-2025
# EDITED   : 1-3-2025
# CONTAINS : Model and helper classes to contain data and information about model
##

from random import randint
from random import shuffle
from math import log2
from typing import Tuple

class Record:

    ## constructor which instantiates both the records 0 total and empty dict
    def __init__(self):
        self.total = 0
        self.dict = {}
    
    ## add_token adds a single token to the record, increasing total and expanding dict as necessary
    def add_token(self, token):
        if token in self.dict:
            self.dict[token] += 1
        else:
            self.dict[token] = 1
        self.total += 1

    ## the non-random prediction method which returns the most likely following token, as well as the confidence of that token out of all tokens
    def predict_next_token(self) -> Tuple[str, float]:
        max_key = max(self.dict, key=self.dict.get)
        return max_key, self.dict[max_key]/self.total

    ## the random prediction methods which returns a random possible token, as well as the confidence of that token out of all tokens.
    def predict_next_token_rand(self) -> Tuple[str, float]:
        x = randint(0, self.total)
        for token in self.dict:
          x -= self.dict[token]
          if x <= 0:
              return token, self.dict[token]/self.total
        return "PIG", 0 #fail state


class Model:

    method_begin_token   = "<beg>"
    method_end_token     = "<end>"
    method_unknown_token = "<unk>"

    ## constructor for Model class initialises n of n-gram, the model brain (lookup-table) as well as empty lists for the data partitioning
    def __init__(self, n : int):
        self.n = n
        self.lookup_table    = {} # keys are token tuples n-1 long
        self.train_data      = []
        self.test_data       = []
        self.validation_data = []

    ## returns an n-1 length sample of data from the test_data
    def get_sample(self) -> list:
        return self.train_data[randint(0, len(self.test_data))]

    ## this method takes a long string of methods separated by "\n" and partitions the data into a split train/test
    def partition_data(self, tokens : list):
        methods = []
        #it's because the beg token is always at idx 0 , and so methods has to be += [1:<end> token idx]
        ind = tokens.index(Model.method_end_token)
        while ind < len(tokens):
            methods += [tokens[1:ind-1]]
            tokens = tokens[ind+1:]
            try:
                ind = tokens.index(Model.method_end_token)
            except:
                break
        shuffle(methods)

        test_indices = 100
        val_indices  = test_indices + int((len(methods) - test_indices) * 0.2)
        self.test_data       = methods[0:test_indices]
        self.validation_data = methods[test_indices:val_indices]
        self.train_data      = methods[val_indices:]
    
    ## copy data partition from another model
    def copy_partition_data(self, model):
        self.train_data      = model.train_data
        self.test_data       = model.test_data
        self.validation_data = model.validation_data

    ## trains model on train data by adding key window to the lookup-table and to the connected record
    def train(self):
        for method in self.train_data:
            window = method[0 : self.n-1]
            for token in method[self.n-1:]:
                hashable_window = tuple(window)
                if hashable_window not in self.lookup_table:
                    self.lookup_table[hashable_window] = Record()
                self.lookup_table[hashable_window].add_token(token)

                window = window[1:] + [token]

    ## evaluates model based on perplexity which is pow(1/product of all prediction probabilities, 1/number of predictions made).
    def eval(self, methods : list) -> float:
        count = 0
        sum_probs = 0
        for method in methods:
            rand_ind = randint(0, len(method) - self.n)
            window = method[rand_ind: rand_ind + self.n-1]
            hashable_window = tuple(window)
            if hashable_window not in self.lookup_table:
                continue
            _, confidence = self.lookup_table[hashable_window].predict_next_token()
            sum_probs += log2(confidence)
            count += 1
        
        if count == 0:
            return 0
        perplexity = pow(2, (-1/count) * sum_probs)
        return perplexity

    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses most likely next token.
    def predict(self, context : list, n = 1000) -> list:
        predicted_tokens = context.copy()
        count = 0
        open_brackets = 0
        hashable_context = tuple(context)
        while hashable_context in self.lookup_table:
            token,_ = self.lookup_table[hashable_context].predict_next_token()
            predicted_tokens += [token]

            if token == "{":
                open_brackets += 1
            if token == "}":
                open_brackets -= 1
                if open_brackets <= 0:
                    return predicted_tokens + [Model.method_end_token]

            context = context[1:] + [token]
            hashable_context = tuple(context)
            
            count += 1
            if count == n:
                return predicted_tokens + [Model.method_end_token]
        return predicted_tokens + [Model.method_unknown_token]
    
    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses probable next token.
    def predict_rand(self, context, n=1000) -> list:
        predicted_tokens = context
        count = 0
        open_brackets = 0
        hashable_context = tuple(context)
        while hashable_context in self.lookup_table:
            token = self.lookup_table[hashable_context].predict_next_token_rand()
            predicted_tokens += [token]

            if token == "{":
                open_brackets += 1
            if token == "}":
                open_brackets -= 1
                if open_brackets <= 0:
                    return predicted_tokens + [Model.method_end_token]

            context = context[1:] + [token]
            hashable_context = tuple(context)
            
            count += 1
            if count == n:
                return predicted_tokens + [Model.method_end_token]
        return predicted_tokens + [Model.method_unknown_token]