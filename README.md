1. GHS REPOS
2. PYDRILLER (.csv) or SrcML(.xml)
3. pre-processing.py
4. tokenization [javalang]
5. make model
6. evaluation based on perplexity
7. training data -> pick the best model
8. test -> perplexity
9. test on a single model

Steps:
- put corpus together (put the two google colab notebooks together locally to run)
- construct big lookup table by processing corpus
- structure of lookup table; dictionary where keys are lists of n-1 tokens, values are sets of tuples containing possible subsequent words and each of their counts
- probabilistic model n-gram: add together all of the counts of the n-1 tokens being looked at - use this
to look at the values of that n-1 tokens key. pick a random number from 0-(the sum), and iterate through tuple values, subtracting each count from that random number. return word in tuple when running count goes negative.
- have the above: as table and count attributes of a Record class