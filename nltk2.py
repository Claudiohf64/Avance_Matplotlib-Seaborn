import spacy

# spacy.cli.download("es_core_news_sm")
nlp = spacy.load("es_core_news_sm")
text = "Hola, ¿cómo estás? hoy aprenderas Procesamiento de Lenguaje Natural"
duc = nlp(text.lower())
print("\n----------Tokenización de palabras con spaCy----------")
print(duc)
for token in duc:
    print(token, token.lemma_)
    
