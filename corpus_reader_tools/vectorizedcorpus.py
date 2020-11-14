from math import log
import corpus_reader_tools.readertools as rt
import pymorphy2

from corpus_reader_tools.pickledcorpusreader import PickledCorpusReader

'''
Класс для tf-idf-векторизации корпуса текста 
и хранения векторизованных данных
'''


class VectorizedCorpus:
    corp_strings = []
    global_entries = []
    local_entries = []
    tfidf_matrix = [[]]
    words_to_index = {}
    docs_count = 0
    _morph = pymorphy2.MorphAnalyzer()

    '''
    Добавляет слово word в словарь words_to_index (если не добалено ранее),
    возвращает words_to_index(key)
    '''

    def get_word_index(self, word: str):
        word = word.lower()
        if not (word in self.words_to_index):
            self.words_to_index[word] = len(self.words_to_index)
        return self.words_to_index[word]

    def vectorize_str_list(self, corp: list):
        self.corp_strings = corp
        for words in corp:
            for word in words:
                self.get_word_index(word)

        words_count = len(self.words_to_index)
        self.docs_count = len(corp)

        self.global_entries = [0] * words_count
        self.local_entries = []

        i = 0
        for doc in corp:
            self.local_entries.append([0.00] * words_count)
            for word in doc:
                if self.local_entries[i][self.get_word_index(word)] == 0:
                    self.global_entries[self.get_word_index(word)] += 1
                self.local_entries[i][self.get_word_index(word)] += 1
            i += 1

        tf_matrix = self.local_entries.copy()

        for row in tf_matrix:
            words_in_doc = len(row) - row.count(0)
            for i in range(0, len(row)):
                try:
                    row[i] = float(row[i]) / float(words_in_doc)
                except ZeroDivisionError:
                    row[i] = 0.00

        self.tfidf_matrix = tf_matrix.copy()
        for row in self.tfidf_matrix:
            for i in range(0, len(row)):
                row[i] *= log(float(self.docs_count) / float(self.global_entries[i]))
        return self.tfidf_matrix

    def vectorize_tagged_corpus(self, corpus: PickledCorpusReader, with_lemma=False, with_pos=False):
        docs = list(corpus.paras())
        new_docs = []
        i = 0
        for doc in docs:
            new_docs.append([])
            for sent in doc:
                for token in sent:
                    if token[1] != 'NONLEX':
                        new_word = ""
                        if with_lemma:
                            new_word += token[2]
                        else:
                            new_word += token[0]
                        if with_pos:
                            new_word += "_" + token[1]
                        new_docs[i].append(new_word)
            i += 1
        return self.vectorize_str_list(new_docs)

    def add_doc(self, s: str):
        self.corp_strings.append(rt.pos_tag_string(s))
        return self.vectorize_str_list(self.corp_strings)

    def get_approx_prediction_vector(self, words):
        words_count = len(self.words_to_index)
        entries = [0.00] * words_count

        for word in words:
            if word.lower() in self.words_to_index:
                entries[self.get_word_index(word)] += 1

        tf_matrix = entries.copy()

        words_in_doc = len(tf_matrix) - tf_matrix.count(0)
        for i in range(0, len(tf_matrix)):
            try:
                tf_matrix[i] = float(tf_matrix[i]) / float(words_in_doc)
            except ZeroDivisionError:
                tf_matrix[i] = 0.00

        tfidf_list = tf_matrix.copy()
        for i in range(0, words_count):
            tfidf_list[i] *= log(float(self.docs_count + 1) / float(self.global_entries[i] + 1))
        return tfidf_list
