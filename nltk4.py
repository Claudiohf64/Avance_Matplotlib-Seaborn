import spacy
nlp = spacy.load("es_core_news_sm")
texto='El director de SENATI publico que se Apertura una oficiona nueva en Puno Carlos Mendez'
print("\n----------Tokenización de palabras con spaCy----------")
doc = nlp(texto.lower())
print(doc)
for token in doc.ents:
    print(token, token.pos_)