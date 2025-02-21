from random import randint
from random import shuffle

class Record:

    def __init__(self):
        self.total = 0
        self.dict = {}
    
    def add_token(self, token):
        if token in self.dict:
            self.dict[token] += 1
        else:
            self.dict[token] = 1
        self.total += 1

    def predict_next_token(self):
        max_key = max(self.dict, key=self.dict.get)
        return max_key, self.dict[max_key]/self.total

    def predict_next_token_rand(self):
        x = randint(0, self.total)
        for token in self.dict:
          x -= self.dict[token]
          if x <= 0:
              return token, self.dict[token]/self.total
        return "PIG" #fail state


class Model:

    def __init__(self, n : int):
        self.n = n
        self.lookup_table = {} # keys are token lists n-1 long
        self.train_data = []
        self.test_data = []

    def set_data(self, tokens : list):
        methods = []
        ind = tokens.index("\n")
        while ind >= 0:
            methods += [tokens[0:ind]]
            tokens = tokens[ind + 1:]
        shuffle(methods)

        split_ind = int(len(methods) * 0.8)
        self.train_data = methods[0:split_ind]
        self.test_data  = methods[split_ind:]

    #train on a list of tokenized methods
    def train(self):
        for method in self.train_data:
            window = method[0 : self.n-1]
            for token in method[self.n-1:]:
                if window not in self.lookup_table:
                    self.lookup_table[window] = Record()
                self.lookup_table[window].add_token(token)

                window = window[1:] + [token]

    #evaluate and return values used to eval model. Using perplexity, which is confidence of the model. Given a starting window
    #based on the test data, what is the confidence of the next generated token.
    def eval(self):
        count = 0
        product_probs = 1
        for method in self.test_data:
            window = method[0:self.n]
            if window not in self.lookup_table:
                continue
            _, confidence = self.lookup_table[window].predict_next_token()
            product_probs *= confidence
            count += 1
        
        perplexity = (1/product_probs)**(1/count)
        return perplexity

    def predict(self, context, n = 1000):
        predicted_tokens = context
        count = 0
        while context in self.lookup_table:
            token = self.lookup_table[context].predict_next_token()
            predicted_tokens += [token]
            context = context[1:] + [token]
            
            count += 1
            if count == n:
                return predicted_tokens
        return predicted_tokens
    
    def predict_rand(self, context, n=1000):
        predicted_tokens = context
        count = 0
        while context in self.lookup_table:
            token = self.lookup_table[context].predict_next_token_rand()
            predicted_tokens += [token]
            context = context[1:] + [token]
            
            count += 1
            if count == n:
                return predicted_tokens
        return predicted_tokens