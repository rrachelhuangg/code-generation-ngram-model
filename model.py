from random import randint

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
        x = randint(0, self.total)
        for token in self.dict:
          x -= self.dict[token]
          if x <= 0:
              return token
        return "PIG" #fail state


class Model:

    def __init__(self, n : int):
        self.n = n
        self.lookup_table = {} # keys are token lists n-1 long

    #train on a list of tokenized methods
    def train(self, tokens):
        for method in tokens:
            window = method[0 : self.n-1]
            for token in method[self.n-1:]:
                if window not in self.lookup_table:
                    self.lookup_table[window] = Record()
                self.lookup_table[window].add_token(token)

                window = window[1:] + [token]

    #evaluate and return values used to eval model
    def eval(self):
        pass

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