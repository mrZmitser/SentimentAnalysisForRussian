import os
import pickle

import nltk
import pymorphy2
from nltk import pos_tag
from nltk.corpus import CategorizedPlaintextCorpusReader
import time
from corpus_reader_tools.pickledcorpusreader import PickledCorpusReader


CAT_PATTERN = r'(phone\_\w+)[^.txt]'
FILE_PATTERN = CAT_PATTERN + r'/\d{2}\.\d{2}.\d{4}\_\d{2}.\d{2}.+\.txt'


def read_corpus(root_dir):
    return CategorizedPlaintextCorpusReader(root_dir, FILE_PATTERN, cat_pattern=CAT_PATTERN)


def read_processed_corpus(root_dir):
    reader = PickledCorpusReader(root_dir, FILE_PATTERN, cat_pattern=CAT_PATTERN)
    return reader


def describe_corp(corp, fileids=None, categories=None):
    started = time.time()
    # Структуры для подсчета.
    counts = nltk.FreqDist()
    tokens = nltk.FreqDist()
    # Выполнить обход абзацев, выделить лексемы и подсчитать их
    counts['paras'] = len(list(corp.paras()))
    counts['sents'] = len(list(corp.sents()))
    for sent in corp.sents():
        for word in sent:
            counts['words'] += 1
            if type(word) == str:
                tokens[word] += 1
            else:
                tokens[word[0]] += 1

    n_fileids = len(corp.fileids())
    n_topics = len(corp.categories())
    # Вернуть структуру данных с информацией
    return {
        'Файлов': n_fileids,
        'Папок': n_topics,
        'Предложений': counts['sents'],
        'Слов': counts['words'],
        'Различных слов и знаков препинания': len(tokens),
        'Разнообразность лексики (слов / различных слов)': float(counts['words']) / float(len(tokens)),
        'Предложений / файл': float(counts['sents']) / float(n_fileids),
        'Время обработки, с': time.time() - started,
    }


def tokenize(corpus, fileid):
    for paragraph in corpus.paras(fileid):
        for sent in paragraph:
            yield pos_tag(sent, lang="rus")


def tokenize_lemmatize(corpus, fileid):
    sents = list(tokenize(corpus, fileid))
    morph = pymorphy2.MorphAnalyzer()
    for sent in sents:
        for i in range(0, len(sent)):
            sent[i] = (sent[i][0], sent[i][1], morph.parse(sent[i][0])[0].normal_form)
    return sents


def process(corpus, target_root_dir, fileid):
    if not os.path.isdir(target_root_dir):
        raise ValueError(
            "Please supply a directory to write preprocessed data to."
        )

    target_dir = target_root_dir + '/' + os.path.dirname(fileid)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_fileid = target_dir + '/' + os.path.basename(fileid)

    document = list(tokenize_lemmatize(corpus, fileid))

    # Записать данные в архив на диск
    with open(target_fileid, 'wb') as f:
        pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)
    del document
    return target_fileid


def transform(corpus: CategorizedPlaintextCorpusReader, target_root_dir):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    open(target_root_dir + "\\meta.info", 'w').write("tagged\nmarks.txt")
    for fileid in corpus.fileids():
        yield process(corpus, target_root_dir, fileid)


def pos_tag_string(s: str):
    sents = nltk.sent_tokenize(s, language="russian")

    for i in range(0, len(sents)):
        sents[i] = nltk.wordpunct_tokenize(sents[i])
        sents[i] = nltk.pos_tag(sents[i], lang="rus")
    new_words = []

    for sent in sents:
        for token in sent:
            if token[1] != 'NONLEX':
                new_words.append(token[0] + "_" + token[1])

    return new_words


def get_meta(directory):
    path = directory + "\\meta.info"
    print(os.path.abspath(path))
    print(os.path.exists(os.path.abspath(path)))
    if not os.path.exists(path):
        return None, None
    f = open(path, 'r')
    strings = f.read().split('\n')
    return strings
