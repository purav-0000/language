import nltk
import sys
import os
import string
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    path_split = directory.split(os.sep)
    new_path = os.path.join(*path_split)

    files = os.listdir(new_path)
    file_to_content = dict()

    for file in files:
        with open(f"{new_path}{os.sep}{file}", encoding='UTF-8') as f:
            content = f.read()
            file_to_content[file] = content

    return file_to_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    document = document.lower()
    document = document.lower()
    document = document.translate(str.maketrans('', '', string.punctuation))

    filtered_document = list()
    words = nltk.word_tokenize(document)

    for w in words:
        if w not in nltk.corpus.stopwords.words('english'):
            filtered_document.append(w)

    return filtered_document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_to_idf = dict()
    counter = 0
    for words in documents.values():

        for word in words:
            counter = 0
            if word not in words_to_idf.keys():

                for document_value in documents.values():
                    if word in document_value:
                        counter += 1
                        continue

                idf = math.log(len(documents)/counter)
                words_to_idf[word] = idf

    return words_to_idf



def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_to_tfidf = dict()
    for file in files:
        file_to_tfidf[file] = 0

    for word in query:
        for file in files.keys():
            if word in files[file]:
                counter = Counter(files[file])
                file_to_tfidf[file] += counter[word] * idfs[word]

    to_be_deleted = list()
    for key in file_to_tfidf.keys():
        if file_to_tfidf[key] == 0:
            to_be_deleted.append(key)

    for item in to_be_deleted:
        del file_to_tfidf[item]
    sorted_files = list()

    file_to_tfidf = sorted(file_to_tfidf.items(), key=lambda x: x[1], reverse=True)

    for item in file_to_tfidf:
        sorted_files.append(item[0])

    n_files = list()

    for i in range(0, FILE_MATCHES):
        n_files.append(sorted_files[i])

    return n_files



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    sentence_to_idf = dict()

    for sentence in sentences:
        sentence_to_idf[sentence] = 0

    for word in query:
        for sentence in sentences.keys():
            if word in sentences[sentence]:
                sentence_to_idf[sentence] += idfs[word]


    to_be_deleted = list()
    for key in sentence_to_idf.keys():
        if sentence_to_idf[key] == 0:
            to_be_deleted.append(key)

    for item in to_be_deleted:
        del sentence_to_idf[item]

    sorted_sentences = list()

    sentence_to_idf = sorted(sentence_to_idf.items(), key=lambda x: (query_words_in_sentence(query, sentences[x[0]])/len(sentences[x[0]])), reverse=True)
    sentence_to_idf = sorted(sentence_to_idf, key=lambda x: x[1], reverse=True)

    for item in sentence_to_idf:
        sorted_sentences.append(item[0])

    n_sentences = list()

    for i in range(0, SENTENCE_MATCHES):
        n_sentences.append(sorted_sentences[i])

    return n_sentences


def query_words_in_sentence(query, sentence):

    counter = 0

    for query_word in query:
        for word_sentence in sentence:
            if query_word == word_sentence:
                counter += 1

    return counter

if __name__ == "__main__":
    main()
