from random import randint
from random import shuffle

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

    ## constructor for Model class initialises n of n-gram, the model brain (lookup-table) as well as empty lists for the data partitioning
    def __init__(self, n : int):
        self.n = n
        self.lookup_table = {} # keys are token lists n-1 long
        self.train_data = []
        self.test_data = []

    ## this method takes a long string of methods separated by "\n" and partitions the data into a split train/test
    def partition_data(self, tokens : list, train = 0.8):
        methods = []
        ind = tokens.index("\n")
        while ind >= 0:
            methods += [tokens[0:ind]]
            tokens = tokens[ind + 1:]
        shuffle(methods)

        split_ind = int(len(methods) * train)
        self.train_data = methods[0:split_ind]
        self.test_data  = methods[split_ind:]

    ## trains model on train data by adding key window to the lookup-table and to the connected record
    def train(self):
        for method in self.train_data:
            window = method[0 : self.n-1]
            for token in method[self.n-1:]:
                if window not in self.lookup_table:
                    self.lookup_table[window] = Record()
                self.lookup_table[window].add_token(token)

                window = window[1:] + [token]

    ## evaluates model based on perplexity which is pow(1/product of all prediction probabilities, 1/number of predictions made).
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

    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses most likely next token.
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
    
    ## makes a continued prediction based on the context until number of predicted tokens reaches n or there are no more predictions. Uses probable next token.
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