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

x = [
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
y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
textos_limpios = [limpiar_tokenizar(t) for t in textos]

x_train, x_test, y_train, y_test = train_test_split(
    textos_limpios, y, test_size=0.3
)

vectorizer = TfidfVectorizer()
x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vect, y_train)
model.fit(x,y,epochs=100)
nuevos = [
    "La compra fue excelente y rápida",
    "Mala calidad del producto, no funciona",
    "No volvería a comprar jamás",
    "Estoy muy satisfecho con el servicio",
]

nuevos_limpios = [limpiar_tokenizar(t) for t in nuevos]
nuevos_vect = vectorizer.transform(nuevos_limpios)
res = model.predict(nuevos_vect)
for i in res:
    print("Texto: ", i , " - Predicción: ", 'Positivo' if i == 1 else 'Negativo')