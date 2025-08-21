from nltk.util import ngrams
from nltk.lm import laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_every gram_pipeline
def ngram_smoothing(sentence,n):
    tokens=word_tokenize(sentence.lower())
    train_data,padded_sents=padded_every gram_pipeline(n,tokens)
    model=laplace(n)
    model.fit(train_data,padded_sents)
    reurn model
sentence=input("Enter a sentence:")
n=int(input("Enter the value of N for N-grams)
