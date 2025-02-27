##  
# AUTHOR   : Rachel Huang, Jackson Taylor
# CREATED  : 20-2-2025
# EDITED   : 27-2-2025
# CONTAINS : Model and helper classes to contain data and information about model
##

from random import randint
from random import shuffle
from math import log2

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
    def predict_next_token(self):
        max_key = max(self.dict, key=self.dict.get)
        return max_key, self.dict[max_key]/self.total

    ## the random prediction methods which returns a random possible token, as well as the confidence of that token out of all tokens.
    def predict_next_token_rand(self):
        x = randint(0, self.total)
        for token in self.dict:
          x -= self.dict[token]
          if x <= 0:
              return token, self.dict[token]/self.total
        return "PIG" #fail state


class Model:

    method_begin_token   = "<beg>"
    method_end_token     = "<end>"
    method_unknown_token = "<unk>"

    ## constructor for Model class initialises n of n-gram, the model brain (lookup-table) as well as empty lists for the data partitioning
    def __init__(self, n : int):
        self.n = n
        self.lookup_table = {} # keys are token lists n-1 long
        self.train_data = []
        self.test_data = []

    ## returns an n-1 length sample of data from the test_data
    def get_sample(self):
        return self.test_data[randint(0, len(self.test_data))][:self.n-1]

    ## this method takes a long string of methods separated by "\n" and partitions the data into a split train/test
    def partition_data(self, tokens : list):
        methods = []
        #it's because the beg token is always at idx 0 , and so methods has to be += [1:<end> token idx]
        ind = tokens.index(Model.method_end_token)
        while ind < len(tokens):
            methods += [tokens[1:ind]]
            tokens = tokens[ind+1:]
            try:
                ind = tokens.index(Model.method_end_token)
            except:
                break
        shuffle(methods)

        split_ind = 100
        self.train_data = methods[split_ind:]
        self.test_data  = methods[0:split_ind]
        return len(self.train_data), len(self.test_data)

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
    def eval(self):
        count = 0
        sum_probs = 0
        for method in self.test_data:
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
        perplexity = pow(2, (-1/self.n) * sum_probs)
        #this is wrong
        return perplexity

    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses most likely next token.
    def predict(self, context : list, n = 1000):
        predicted_tokens = context.copy()
        count = 0
        hashable_context = tuple(context)
        while hashable_context in self.lookup_table:
            token,_ = self.lookup_table[hashable_context].predict_next_token()
            predicted_tokens += [token]

            if token == Model.method_end_token:
                return predicted_tokens

            context = context[1:] + [token]
            hashable_context = tuple(context)
            
            count += 1
            if count == n:
                return predicted_tokens + [Model.method_end_token]
        return predicted_tokens + [Model.method_unknown_token]
    
    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses probable next token.
    def predict_rand(self, context, n=1000):
        predicted_tokens = context
        count = 0
        hashable_context = tuple(context)
        while hashable_context in self.lookup_table:
            token = self.lookup_table[hashable_context].predict_next_token_rand()
            predicted_tokens += [token]

            if token == Model.method_end_token:
                return predicted_tokens

            context = context[1:] + [token]
            hashable_context = tuple(context)
            
            count += 1
            if count == n:
                return predicted_tokens + [Model.method_end_token]
        return predicted_tokens + [Model.method_unknown_token]