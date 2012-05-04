from features import Feature, TrainedFeature
from models import LassoPathSelector
try:
    import gensim
except ImportError:
    pass
from utils import bag_of_words, cosine, tokenize, tokenize_keep_all, tokenize_with_sentinels
from pandas import DataFrame, read_csv, concat, Series
from core import get_data, set_data, load_object, save_object, get_setting
import hashlib
import re
import numpy as np
import math
try:
    import nltk
except ImportError:
    pass

debug = False



sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def make_docs_hash(docs):
    m = hashlib.md5()
    for doc in docs:
        for tok in doc:
            try:
                m.update(tok)
            except UnicodeEncodeError:
                pass
    return m.hexdigest()

#TODO: no more storable... :(
class Dictionary(object):
    def __init__(self, mindocs=3, maxterms=100000, maxdocs=.9, force=False):
        self.mindocs = mindocs
        self.maxterms = maxterms
        self.maxdocs = maxdocs
        self.dictionary = gensim.corpora.Dictionary
        self.dataset = None
        self.force = force
        self.storable_hash = self._make_hash(
                mindocs,
                maxterms,
                maxdocs)

    def name(self, docs, type_='dict'):
        return '%s_%s' % (make_docs_hash(docs), type_)

    def get_dict(self, dataset, docs):
        try:
            dct = self.dataset.store.load(self.name(docs))
            return dct
        except IOError:
            return self._make_dict(docs)

    def _make_dict(self, docs):
        dct = self.dictionary(docs)
        dct.filter_extremes(no_below=self.mindocs, no_above=self.maxdocs,
                keep_n=self.maxterms)
        # dct.save(self.data_dir + self.name(docs))
        self.dataset.store.save(self.name(docs), dct)
        return dct

    def get_tfidf(self, docs):
        try:
            return self.dataset.store.load(
                    self.name(docs, 'tfidf'))
        except IOError:
            return self._make_tfidf(docs)

    def _make_tfidf(self, dataset, docs):
        dct = self.get_dict(dataset, docs)
        # corpus = [dct.doc2bow(d) for d in docs]
        tfidf = gensim.models.TfidfModel(dictionary=dct)
        self.dataset.store.save(self.name(docs, 'tfidf'), tfidf)
        return tfidf




class TopicModelFeature(Feature):
    def __init__(self, feature, topic_modeler=None, num_topics=50, force=False,
            stored_model=None, mindocs=3, maxterms=100000, maxdocs=.9,
            tokenizer=tokenize):
        self.mindocs = mindocs
        self.maxterms = maxterms
        self.maxdocs = maxdocs
        super(TopicModelFeature, self).__init__(feature)
        self.num_topics = num_topics
        self.dictionary = gensim.corpora.Dictionary
        self.tokenizer = tokenizer
        self.topic_modeler = topic_modeler
        self.stored_model = stored_model
        self.data_dir = get_setting('DATA_DIR')
        self.force = force
        self._name = '%s_%d' %(self._name, num_topics)
        # self._hash = self._make_hash(
        #         num_topics,
        #         mindocs,
        #         maxterms,
        #         maxdocs)

    def make_docs_hash(self, docs):
        m = hashlib.md5()
        for doc in docs:
            for tok in doc:
                try:
                    m.update(tok)
                except UnicodeEncodeError:
                    pass
        return m.hexdigest() + '_' + str(self.num_topics)


    def _create(self, data):
        vecs = None #self.load('topic_vecs')
        if vecs is None or self.force:
            vecs = self.make_vectors(data)
        vecs.columns = ['%s_%s'%(c, data.name) for c in vecs.columns]
        return vecs

    def make_docs(self, data):
        docs = [self.tokenizer(d) for d in data]
        print "docs", docs[0][:80]
        self._docs_hash = self.make_docs_hash(docs)
        return docs

    def make_dict(self, docs):
        dct = self.dictionary(docs)
        dct.filter_extremes(no_below=self.mindocs, no_above=self.maxdocs,
                keep_n=self.maxterms)
        dct.save(self.data_dir + self.dict_name)
        return dct


    def make_engine(self, docs):
        print "making vectors"
        dct = self.make_dict(docs)
        corpus = [dct.doc2bow(d) for d in docs]
        #corpora.MmCorpus.serialize('history/data/%s_token.mm'%self.name, corpus)
        #for transform in transforms
        tfidf = gensim.models.TfidfModel(corpus)
        tfidf.save(self.data_dir + self.tfidf_name)
        topic_model = self.topic_modeler(corpus=tfidf[corpus], id2word=dct,
                num_topics=self.num_topics)
        print topic_model
        topic_model.save(self.data_dir + self.topic_model_name)
        return self.load_engine()



    @property
    def vectors_name(self):
        return self.make_key('vecs')
    @property
    def dict_name(self):
        return self._docs_hash + '-dict'
    @property
    def tfidf_name(self):
        return self._docs_hash + '-tfidf'
    @property
    def topic_model_name(self):
        return self._docs_hash + self.topic_modeler.__name__

    def load_engine(self):
        if self.stored_model:
            dct = self.dictionary.load(self.data_dir + self.stored_model[0])
            tfidf = gensim.models.TfidfModel.load(self.data_dir + self.stored_model[1])
            lsi = self.topic_modeler.load(self.data_dir + self.stored_model[2])
            print "using stored model"
        else:
            dct = self.dictionary.load(self.data_dir + self.dict_name)
            tfidf = gensim.models.TfidfModel.load(self.data_dir + self.tfidf_name)
            lsi = self.topic_modeler.load(self.data_dir + self.topic_model_name)
        return (dct, tfidf, lsi)
    def label(self, tokens):
        return self.dictionary.doc2bow(tokens)

    def transform(self, vect):
        for transform in self.transforms:
            vect = transform[vect]
        return vect

    def get_vector_for_text(self, txt):
        return self.transform(self.label(self.tokenizer(txt)))

    def make_vectors(self, ds, n=None):
        docs = self.make_docs(list(ds.values))
        try:
            if self.force:
                dct, tfidf, lsi = self.make_engine(docs)
            else:
                dct, tfidf, lsi = self.load_engine()
        except IOError:
            dct, tfidf, lsi = self.make_engine(docs)
        vecs = []
        print "Making topic vectors"
        for i, txt in enumerate(ds):
            topic_vec = dict(
                    lsi[tfidf[dct.doc2bow(
                        self.tokenizer(txt))]]
                    )
            if not topic_vec:
                print "blankkkk topic vector"
            if len(topic_vec) != self.num_topics:
                print "short vector"
                missing = set(range(self.num_topics)) - set(topic_vec.keys())
                print "filling with zero for %d topics" %len(missing)
                for k in missing:
                    topic_vec[k] = 0
            vecs.append(topic_vec)
        tvecs = DataFrame(vecs, index=ds.index)
        return tvecs

class LSI(TopicModelFeature):
    def __init__(self, *args, **kwargs):
        kwargs['topic_modeler'] = gensim.models.lsimodel.LsiModel
        super(LSI, self).__init__(*args, **kwargs)

class SentenceLSI(TopicModelFeature):
    def __init__(self, *args, **kwargs):
        kwargs['topic_modeler'] = gensim.models.lsimodel.LsiModel
        super(SentenceLSI, self).__init__(*args, **kwargs)

    def make_docs(self, data):
        sents = []
        for txt in data:
            sents.extend(sent_tokenizer.tokenize(txt))
        docs = [self.tokenizer(d) for d in sents]
        print "docs", docs[0][:20]
        self._docs_hash = self.make_docs_hash(docs)
        return docs


class LDA(TopicModelFeature):
    def __init__(self, *args, **kwargs):
        kwargs['topic_modeler'] = gensim.models.ldamodel.LdaModel
        super(LDA, self).__init__(*args, **kwargs)


class TFIDF(Feature):
    def __init__(self, feature, mindocs=50, maxterms=10000, maxdocs=1.):
        super(TFIDF, self).__init__(feature)
        # self.mindocs = mindocs
        # self.maxterms = maxterms
        # self.maxdocs = maxdocs
        # self.data_dir = get_setting('DATA_DIR')
        # self.force = force
        # self.tokenizer = tokenize
        self.dictionary = Dictionary(mindocs, maxterms, maxdocs)
        # self._hash = self._make_hash(
        #         mindocs,
        #         maxterms,
        #         maxdocs)

    def _create(self, data):
        docs = list(data)
        dct = self.dictionary.get_dict(self.dataset, docs)
        tfidf = self.dictionary.get_tfidf(self.dataset, docs)
        docs = [dct.doc2bow(d) for d in docs]
        vecs = tfidf[docs]
        df = DataFrame([dict(row) for row in vecs], index=data.index)
        df.columns = ['%s_%s' % (dct[i], data.name) for i in df.columns]
        df = df.fillna(0)
        print df
        return df

class NgramCounts(Feature):
    def __init__(self, feature, mindocs=50, maxterms=10000, maxdocs=1.):
        super(NgramCounts, self).__init__(feature)
        # self.mindocs = mindocs
        # self.maxterms = maxterms
        # self.maxdocs = maxdocs
        # self.data_dir = get_setting('DATA_DIR')
        # self.force = force
        # self.tokenizer = tokenize
        self.dictionary = Dictionary(mindocs, maxterms, maxdocs)
        # self._hash = self._make_hash(
        #         mindocs,
        #         maxterms,
        #         maxdocs)

    def _create(self, data):
        docs = list(data)
        dct = self.dictionary.get_dict(self.dataset, docs)
        docs = [dct.doc2bow(d) for d in docs]
        df = DataFrame([dict(row) for row in docs], index=data.index)
        df.columns = ['%s_%s' % (dct[i], data.name) for i in df.columns]
        df = df.fillna(0)
        return df



import nltk
class TreebankTokenize(Feature):
    tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    def _create(self, data):
        return data.map(self.tokenizer.tokenize)

def ngrams(toks, n, sep='|'):
    return [sep.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
class Ngrams(Feature):
    def __init__(self, feature, ngrams=1):
        self.ngrams = ngrams
        super(Ngrams, self).__init__(feature)
        self._name += '_%d'%ngrams

    def _create(self, data):
        return data.map(lambda x: ngrams(x, self.ngrams))


class Tokenizer(Feature):
    def __init__(self, feature, tokenizer=tokenize_keep_all):
        super(Tokenizer, self).__init__(feature)
        self.tokenizer = tokenizer

    def _create(self, data):
        return data.map(self.tokenizer)

import syllables
class SyllableCount(Feature):
    def _create(self, data):
        return data.map(lambda txt: sum([syllables.count(w) for w in txt.split()]))

def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / float(len(a | b))

class ClosestDoc(Feature):
    def __init__(self, feature, text, doc_splitter=sent_tokenizer.tokenize,
            tokenizer=tokenize, sim=jaccard):
        super(ClosestDoc, self).__init__(feature)
        self.text = text
        self.tokenizer = tokenizer
        self.doc_splitter=doc_splitter
        self.sim = sim

    def make_docs(self, data):
        docs = []
        for d in data:
            docs.append(self.doc_splitter(d))
        return data.map(
                lambda x: [self.tokenizer(d) for d in self.doc_splitter(x)])

    def score(self, data):
        scores = []
        txt = self.tokenizer(self.text)
        for d in self.make_docs(data):
            score = max([self.sim(txt, doc) for doc in d])
            scores.append(score)
        return scores

    def _create(self, data):
        return Series(self.score(data), index=data.index, name=data.name)

# chkr = SpellChecker("en_US")
from aspell import Speller
chkr = Speller()
def count_spell_errors(toks, exemptions):
    if not toks:
        return 0
    return sum([not chkr.check(t) for t in toks if t not in exemptions])
class SpellingErrorCount(Feature):
    def __init__(self, feature, exemptions=None):
        super(SpellingErrorCount, self).__init__(feature)
        self.exemptions = exemptions if exemptions else set()

    def _create(self, data):
        return data.map(lambda x : count_spell_errors(x, self.exemptions))

spelling_suggestions = {}

import collections
def words(text): return re.findall("[a-z']+", text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


wordlist = set(nltk.corpus.words.words('en'))
from nltk.corpus import wordnet

def is_nondict(t):
    return not wordnet.synsets(t) and t not in wordlist

def nondict_w_exemptions(toks, exemptions, count_typos=False):
    cnt = 0
    for t in toks:
        if is_nondict(t) and t not in exemptions:
            if not count_typos:
                sug = suggest_correction(t)
                if sug.replace(' ', '') == t: continue
            cnt += 1
    return cnt

class NonDictCount(Feature):
    def __init__(self, feature, exemptions=None):
        super(NonDictCount, self).__init__(feature)
        self.exemptions = exemptions if exemptions else set()

    def _create(self, data):
        return data.map(lambda toks: nondict_w_exemptions(toks, self.exemptions))

class RemoveNonDict(Feature):
    """Expects tokens"""
    def _create(self, data):
        return data.map(
                lambda toks: [t for t in toks if not wordnet.synsets(t) and t not in wordlist]
                )

def expanded_tokenize(s):
    toks = []
    for tok in tokenize(s):
        for syn in wordnet.synsets(tok):
            toks.extend(syn.lemma_names)
    return toks

class ExpandedTokens(Feature):

    def _create(self, data):
        return data.map(expanded_tokens)

class LongwordCount(Feature):
    def __init__(self, feature, lengths=[6, 7, 8]):
        super(LongwordCount, self).__init__(feature)
        self.lengths = lengths
    def _create(self, data):
        cols = []
        for length in self.lengths:
            cols.append(data.map(lambda tokens: sum([len(t) > length for t in
            tokens])))
        return concat(cols, keys=['%s_%s'%(str(w), data.name) for w in self.lengths],
                axis=1)


class SentenceCount(Feature):
    def _create(self, data):
        return data.map(
                lambda x: len(sent_tokenizer.tokenize(x)) + 1)

class SentenceSlice(Feature):
    def __init__(self, feature, start=0, end=None):
        super(SentenceSlice, self).__init__(feature)
        self.start = start
        self.end = end

    def _create(self, data):
        if self.end is None:
            return data.map(
                    lambda x: ' '.join(sent_tokenizer.tokenize(x)[self.start:])
                    )
        return data.map(
                lambda x: ' '.join(sent_tokenizer.tokenize(x)[self.start:self.end])
                )

class SentenceLength(Feature):
    def _create(self, data):
        return data.map(
                lambda x: float(len(x))/(len(sent_tokenizer.tokenize(x)) + 1))

class CapitalizationErrors(Feature):
    def _create(self, data):
        return data.map(
                lambda x: len([1 for s in sent_tokenizer.tokenize(x) if
                    s and s[0]!=s[0].upper() or ' i ' in s]))

class LongestSentence(Feature):
    def _create(self, data):
        return data.map(
                lambda x: max(map(len, sent_tokenizer.tokenize(x)))
                )

class LongestWord(Feature):
    def _create(self, tokens):
        return tokens.map(
                lambda x: max([len(t) for t in x])
                )

class VocabSize(Feature):
    def _create(self, data):
        return data.map(lambda tokens: len(set(tokens)))

class KeywordCount(Feature):
    def __init__(self, feature, words):
        super(KeywordCount, self).__init__(feature)
        self.words = set(words)
    def _create(self, data):
        cols = []
        for word in self.words:
            cols.append(data.map(lambda tokens: tokens.count(word)))
        return concat(cols, keys=[ '%s_%s'%(w, data.name) for w in self.words],
                axis=1)

from char_freqs import char_freqs
from collections import Counter
def char_kl(txt):
    if not txt:
        return 0
    # all caps case... not sure what to do
    if txt.upper() == txt:
        return 1
    c = Counter(txt)
    kl = 0
    tot = float(len(txt))
    eps = 1.0e-05
    for ch, p in char_freqs.items():
        if ch not in c:
            q = eps
        else:
            q = c[ch]/tot
        try:
            kl += p * math.log(p/q)
        except:
            print p, q, c, txt
    return kl


class CharFreqKL(Feature):
    def _create(self, data):
        return data.map(char_kl)

class WeightedWordCount(Feature):
    def _create(self, tokens):
        alltoks = reduce(lambda x,y:x+y, tokens)
        n = float(len(alltoks))
        c = Counter(alltoks)
        return tokens.map(
                lambda toks: sum([1-c[t]/n for t in toks])
                )


class NgramCompare(NgramCounts):
    def __init__(self, feature, *args, **kwargs):
        kwargs['mindocs'] = 1
        kwargs['maxdocs'] = 1
        kwargs['maxterms'] = 1000000
        super(NgramCompare, self).__init__(feature, *args, **kwargs)
        self.tokenizer = tokenize_with_sentinels

    def _create(self, data):
        raw_docs = [s for s in nltk.corpus.gutenberg.raw().split('[') if len(s) > 10000]
        docs = self.make_docs(raw_docs, 1)
        dct1 = self.dictionary(docs)
        dct1.filter_extremes(no_below=1, no_above=1.,
                keep_n=2000)
        print dct1
        n = 2
        docs = [self.tokenizer(d) for d in raw_docs]
        docs = [[self.sep.join(toks[i:i+n]) for i in range(len(toks)-n+1) if
            all([t in dct1.token2id for t in toks[i:i+n]])] for toks in docs]
        print "docs", docs[0][:80]
        dct = self.dictionary(docs)
        docs = [self.tokenizer(d) for d in data]
        docs = [[self.sep.join(toks[i:i+n]) for i in range(len(toks)-n+1) if
            all([t in dct1.token2id for t in toks[i:i+n]])] for toks in docs]
        print "docs", docs[0][:80]
        print "unkown ngrams", [t for t in docs[0] if t not in
                dct.token2id][:200]
        cnts = [sum([t not in dct.token2id for t in toks]) for toks in docs]
        return Series(cnts, index=data.index)
