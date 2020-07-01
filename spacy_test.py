import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Some text read from 95 papers.'
doc = nlp(text)
for w in doc:
    print(w, w.lemma_, w.tag_, w.shape_)

"""
Some some 90 DT Xxxx True False False False
text text 92 NN xxxx True False False False
read read 100 VBD xxxx True False False False
from from 85 IN xxxx True False False False
95 95 93 CD dd False False True True
paper paper 92 NN xxxx True False False False
. . 97 . . False True False False
"""
