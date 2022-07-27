# Aca ponemos nuestro pipeline para correr en este archivo

import pandas as pd 
import regex as reg
import re
import matplotlib.pyplot as plt
import unicodedata
import nltk
import string
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection, svm
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

# Leemos el archivo a tratar
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

# Pasamos los valores de is_spam a 1 si es True y 0 si es False
df_raw['is_spam'] = df_raw['is_spam'].apply(lambda x: 1 if x == True else 0)

# Eliminamos los duplicados

df_raw = df_raw.drop_duplicates()
df_raw = df_raw.reset_index(inplace = False)[['url','is_spam']]

# Empezamos el proceso de limpieza de datos

# colocar todos los textos en minusculas
df_raw['url'] = df_raw['url'].str.lower()

df_aux = df_raw.copy()

# Proceso de limpieza de datos

limpieza = []

for p in range(len(df_aux.url)):
    desc = df_aux['url'][p]
    
    #sacar la puntuacion
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    #borrar etiquetas especiales
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    #borrar digitos y caracteres especiales
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    limpieza.append(desc)

#asiganmos los datos limipios en limpieza
df_aux['url'] = limpieza

# Creamos una lista de stopwords
stop_words = ['http','www','com','you','your','for','not','have','is','in','im','from','to','https','e','c','v','b','f','p']

# Le agregamos al vector de stop_words las palabras de menos de 3 letras para luego sacarlas

for i in df_aux['url'].str.split(expand=True).stack().value_counts().index:
    if len(i)<3 :
        stop_words.append(i)

# con esto de poner set borramos los duplicados
stop_words=list(set(stop_words))

#funcion para borrar los stop_words
def remove_stopwords(message):
  if message is not None:
    words = message.strip().split()
    words_filtered = []
    for word in words:
      if word not in stop_words:
        words_filtered.append(word) 
    result = " ".join(words_filtered)         
  else:
    result = None

  return result 

  # Borramos los stop_words
  df_aux['url']=df_aux['url'].apply(remove_stopwords)

  # Ahora que hemos limpiado nuestros datos
# procedemos a hacer una copia con el dataset a trabajar en el final
df = df_aux.copy()

X = df['url']
y = df['is_spam']

#Creando la Matriz

message_vectorizer = CountVectorizer().fit_transform(df['url'])

# Haciendo el split de los datos

X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['is_spam'], test_size = 0.2, random_state = 121, shuffle = True)

cl = svm.SVC(C=1.0, kernel='linear', degree=4, gamma='auto')

cl.fit(X_train, y_train)

pred = cl.predict(X_train)
print(classification_report(y_train, pred))

pred = cl.predict(X_test)
print(classification_report(y_test, pred))

#Grabamos nuestro modelo 
pickle.dump(cl, open('../models/texto_NLP.pkl', 'wb'))

print("Se termino el proceso correctamente")
print("El archivo se guardo en la carpeta models en forma correcta")
print()
print("===============   Programa  Terminado  ========================")


        