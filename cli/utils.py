import json
import string
from nltk.stem import PorterStemmer


def build_command(inverted):
    inverted.build()
    inverted.save()

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def stop_words():
    with open("data/stopwords.txt", 'r') as file:
        stopwords = file.read().splitlines()
    return list(filter(lambda x: clean_punctuation(x), stopwords))


def check_common(list1, list2):
    if not list1 or not list2:
        return False

    query_words = set(list1)
    title_words = set(list2)
    return any(q_tok in t_tok for q_tok in query_words for t_tok in title_words)


def clean_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    cleaned_string = input_string.translate(translator).lower()
    return cleaned_string


def clean_punctuation_stopwords(string):
    clean = clean_punctuation(string).split()
    stop = stop_words()
    tokens = list(filter(lambda x: x not in stop, clean))
    return stemming(tokens)


def load_file(file_path):
    with open(file_path, 'r') as file:
        movies = json.load(file)["movies"]
    return movies


def search_movies(query, inverted):
    if inverted.load() is None:
        print("Files don't exist. Run build first.")
        return
    tokens = clean_punctuation_stopwords(query)
    found_movies = []
    limit = 5
    for token in tokens:
        if token in inverted.index:
            for doc_id in inverted.index[token]:
                found_movies.append(doc_id)
                if len(found_movies) >= limit:
                    break
        if len(found_movies) >= limit:
            break
    for doc_id in found_movies:
        movie = inverted.docmap[doc_id]
        print(f"{movie['title']} ({doc_id})")