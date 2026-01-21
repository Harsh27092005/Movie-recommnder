#data preprocessing
import numpy as np
import pandas as pd
movies=pd.read_csv("/Users/harshsingh/Desktop/mini project/tmdb_5000_movies.csv")
credits=pd.read_csv("/Users/harshsingh/Desktop/mini project/tmdb_5000_credits.csv")
movies = movies.merge(credits,on='title')
#genres
#id
#keyword
#title
#overview
#cast
#crew(because there are directors in it)
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.iloc[0].genres
import ast
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L
movies.dropna(inplace=True)

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert3)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

#vectrization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Initialize CountVectorizer and PorterStemmer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()

# Apply stemming (if not already applied during preprocessing)
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new['tags'] = new['tags'].apply(stem)

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    for i in sorted_movies:
        print(new.iloc[i[0]].title)


recommend('Gandhi')


