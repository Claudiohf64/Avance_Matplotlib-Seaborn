# 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')
text='Hola, ¿cómo estás? hoy aprenderas Procesamiento de Lenguaje Natural'

# Tokenización de palabras con word_tokenize
tokens = word_tokenize(text=text.lower(), language='spanish')
print("\n----------Tokenización de palabras con word_tokenize----------")
print("con word_tokenize: ",tokens)

# Tokenización de palabras con TweetTokenizer
tokenizer = TweetTokenizer()
tokens = tokenizer.tokenize(text=text.lower())
print("\n----------Tokenización de palabras con TweetTokenizer----------")
print("con TweetTokenizer: ",tokens)

# Eliminación de stopwords
stop_words = set(stopwords.words('spanish'))
filt_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
print ("\n----------Eliminación de stopwords----------")
text_clean=[A for A in filt_tokens if A.isalpha() and A not in stop_words]
print("Sin stopwords: ",text_clean)