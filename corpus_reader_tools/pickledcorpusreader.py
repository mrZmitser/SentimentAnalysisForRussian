import pickle

from nltk.corpus import CategorizedPlaintextCorpusReader

CAT_PATTERN = r'(phone\_\w+)[^.txt]'
FILE_PATTERN = CAT_PATTERN + r'/\d{2}\.\d{2}.\d{4}\_\d{2}.\d{2}.+\.txt'


class PickledCorpusReader(CategorizedPlaintextCorpusReader):
    def __init__(self, root, fileids=FILE_PATTERN, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        CategorizedPlaintextCorpusReader.__init__(self, root, fileids, cat_pattern=kwargs['cat_pattern'])

    def paras(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def sents(self, fileids = None, categories = None):
        for para in self.paras(fileids, categories):
            for sent in para:
                yield sent

    def tokens(self, fileids=None, categories=None):
        for sent in self.sents(fileids, categories):
            for tagged_token in sent:
                yield tagged_token

    def words(self, fileids=None, categories=None):
        for tagged in self.tokens(fileids, categories):
            yield tagged[0]
