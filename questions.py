import nltk
import sys
import os
import string
import math

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
    output = dict()
    for filename in os.listdir(directory):
        output[filename] = ""
        path = f"{directory}{os.sep}{filename}"
        with open(path, encoding = "utf-8") as file:
            for line in file:
                output[filename] += line
    
    return output


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    output = nltk.word_tokenize(document)
    words_to_remove = []
    for word in output:
        if word in string.punctuation:
            words_to_remove.append(word)
        if word in nltk.corpus.stopwords.words("english"):
            words_to_remove.append(word)
    for word in words_to_remove:
        output.remove(word)
    
    return output

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    output = dict()
    for document in documents:
        words_already_seen = []
        for word in documents[document]:
            if word not in output and word not in words_already_seen:
                output[word] = 1
                words_already_seen.append(word)
            elif word in output and word not in words_already_seen:
                output[word] += 1
                words_already_seen.append(word)
    num_documents = len(list(documents.keys()))
    for word in output:
        output[word] = math.log(num_documents / output[word])
    return output




def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = dict() #maps file to sum of tf_idf value
    for file in files:
        tf_idf[file] = 0

    for file in files:
        for word in query:
            tf = files[file].count(word)
            idf = idfs[word]
            if tf != 0:
                tf_idf[file] += tf * idf
    
    output = []
    sorted_tfidf = sorted(list(tf_idf.values()), reverse=True)

    for num in range(n):
        for file in tf_idf:
            if tf_idf[file] == sorted_tfidf[0]:
                output.append(file)
                sorted_tfidf.pop(0)
    #print(output)
    return output

      



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    #Maps sentence to list of [idf, query term density]
    qtd = dict()
    for sentence in sentences:
        qtd[sentence] = [0,0]

    for sentence in sentences:
        for word in query:
            if word in sentences[sentence]:
                qtd[sentence][0] += idfs[word]
    
    sentence_ranking = sorted(list(qtd.keys()), key = lambda x: qtd[x][0], reverse=True)
    
    for sentence in sentences:
        count = 0
        for word in query:
            if word in sentences[sentence]:
                count += 1
        density = count / len(sentences[sentence])
        qtd[sentence][1] = density
    
    max_idf = qtd[sentence_ranking[0]][0]
    sentences_with_highest_idf = []
    for sentence in sentence_ranking:
        if qtd[sentence][0] == max_idf:
            sentences_with_highest_idf.append(sentence)
    
    if len(sentences_with_highest_idf) > 1:
        sentences_with_highest_idf = sorted(sentences_with_highest_idf, key = lambda x: qtd[x][1], reverse = True)
    
    output = []
    if n < len(sentences_with_highest_idf):
        for x in range(n):
            output.append(sentences_with_highest_idf[x])
    else:
        output += sentences_with_highest_idf[x]
        while len(output) != n:
            for sentence in sentence_ranking:
                if sentence not in output:
                    output.append(sentence)
    
    return output
    
if __name__ == "__main__":
    main()
