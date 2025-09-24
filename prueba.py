import spacy

# convierte las palabras a palabras vectoricas ponderadas (limpia el texto)
from sklearn.feature_extraction.text import TfidfVectorizer
# clasificador de texto
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# spacy.download("es_core_news_sm")
nlp = spacy.load("es_core_news_sm")


def limpiar_tokenizar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct]
    return " ".join(tokens)


# texto = 'El director de SENATI publico que se Apertura una oficiona nueva en Puno Carlos Mendez'
# tokens=limpiar_tokenizar(texto)
# print(tokens)

textos = [
    "Me encanta este producto",
    "Es horrible, no lo recomiendo",
    "Excelente atención al cliente",
    "La experiencia fue muy mala",
    "Estoy feliz con la compra",
    "No me gustó para nada",
    "Fue una buena compra",
    "Una pésima decisión",
    "El servicio fue excelente",
    "No volveré jamás",
    "No me gustó nada el servicio",
    "Jamás volvería a comprar aquí",
    "Nada de lo prometido fue cumplido",
    "Muy lento y poco profesional",
    "Estoy muy satisfecho con el servicio",
    "Fue una experiencia muy agradable",
]
etiquetas = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
textos_limpios = [limpiar_tokenizar(t) for t in textos]
print(textos_limpios)