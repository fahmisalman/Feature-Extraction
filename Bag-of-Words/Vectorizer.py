import math


def tf(data):
    words = bag_of_words(data)
    term = []
    for i in range(len(data)):
        temp = []
        for j in range(len(words)):
            temp.append(data[i].count(words[j]))
        term.append(temp)
    return term


def idf(data):
    words = bag_of_words(data)
    doc = []
    for i in range(len(words)):
        df = 0
        for j in range(len(data)):
            if words[i] in data[j]:
                df += 1
        temp = len(data)/df
        doc.append(math.log(temp))
    return doc


def bag_of_words(data):
    words = []
    for i in range(len(data)):
        words += data[i]
    words = list(set(words))
    return words


def tf_idf(data):
    words = bag_of_words(data)

    # TF
    tfreq = []
    for i in range(len(data)):
        temp = []
        for j in range(len(words)):
            temp.append(data[i].count(words[j]))
        tfreq.append(temp)

    # IDF
    idfreq = []
    for i in range(len(words)):
        df = 0
        for j in range(len(data)):
            if words[i] in data[j]:
                df += 1
        temp = len(data) / df
        idfreq.append(math.log(temp))

    # TF-IDF
    tfidf = []
    for i in range(len(tfreq)):
        temp = []
        for j in range(len(idfreq)):
            temp.append(tfreq[i][j] * idfreq[j])
        tfidf.append(temp)
    return tfidf, words


def binary(data):
    words = bag_of_words(data)
    term = []
    for i in range(len(data)):
        temp = []
        for j in range(len(words)):
            if words[j] in data[i]:
                temp.append(1)
            else:
                temp.append(0)
        term.append(temp)
    return term
