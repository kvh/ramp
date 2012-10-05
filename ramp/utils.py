#from history.models import *
import urlparse
import re
from BeautifulSoup import BeautifulStoneSoup
import numpy as np
from hashlib import md5

def _pprint(params, offset=0, printer=repr):
    # Jacked from scikit-learn
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params: dict
        The dictionary to pretty print

    offset: int
        The offset in characters to add at the begin of each line.

    printer:
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    #line_sep = ',\n' + (1 + offset // 2) * ' '
    line_sep = ','
    for i, (k, v) in enumerate(sorted(params.iteritems())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        # if len(this_repr) > 500:
        #     this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75
                                        or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(',')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
import random
def make_folds(index, nfolds=5, repeat=1, shuffle=True):
    n = len(index)
    indices = range(n)
    foldsize = n / nfolds
    for i in range(repeat):
        if shuffle:
            random.shuffle(indices)
        for i in range(nfolds):
            test = index[indices[i*foldsize:(i + 1)*foldsize]]
            train = index - test
            assert not (train & test)
            yield train, test

def get_np_hash(obj):
    hshr = md5
    try:
        return hshr(np.getbuffer(obj)).hexdigest()
    except TypeError:
        # Cater for non-single-segment arrays: this creates a
        # copy, and thus aleviates this issue.
        # XXX: There might be a more efficient way of doing this
        return hshr(np.getbuffer(obj.flatten())).hexdigest()

def get_np_hashable(obj):
    try:
        return np.getbuffer(obj)
    except TypeError:
        return np.getbuffer(obj.flatten())


def get_single_column(df):
    assert len(df.columns) == 1
    return df[df.columns[0]]


re_object_repr = re.compile(r'<([.a-zA-Z0-9_ ]+?)\sat\s\w+>')

def stable_repr(obj):
    state = _pprint(obj.__getstate__())
    # HACK: replace 'repr's that contain object id references
    state = re_object_repr.sub(r'<\1>', state)
    return '%s(%s)' % (
            obj.__class__.__name__,
            state)

stop_words = set([
    'http',
    'https',
    'com',
    'www',
    'org',
    'web',
    'url',
    'the',
    'a',
    'and',
    'of',
    'it',
'i'      ,
'a',
'about' ,
'an' ,
'are' ,
'as' ,
'at' ,
'be' ,
'by' ,
'com' ,
'for' ,
'from',
'how',
'in' ,
'is' ,
'it' ,
'of' ,
'on' ,
'or' ,
'that',
'the' ,
'this',
'to' ,
'was' ,
'what' ,
'when',
'where',
'who' ,
'will' ,
'with',
'the',
'www'
    ])

contractions = {"aren't":"are not",
"can't":"cannot",
"couldn't":"could not",
"didn't":"did not",
"doesn't":"does not",
"don't":"do not",
"hadn't":"had not",
"hasn't":"has not",
"haven't":"have not",
"he'd":"he had",
"he'll":"he will",
"he's":"he is",
"i'd":"i had",
"i'll":"i will",
"i'm":"i am",
"i've":"i have",
"isn't":"is not",
"it'll":"it will",
"it's":"it is",
"let's":"let us",
"mightn't":"might not",
"mustn't":"must not",
"shan't":"shall not",
"she'd":"she had",
"she'll":"she will",
"she's":"she is",
"shouldn't":"should not",
"that's":"that is",
"there's":"there is",
"they'd":"they had",
"they'll":"they will",
"they're":"they are",
"they've":"they have",
"we'd":"we had",
"we're":"we are",
"we've":"we have",
"weren't":"were not",
"what'll":"what will",
"what're":"what are",
"what's":"what is",
"what've":"what have",
"where's":"where is",
"who'd":"who had",
"who'll":"who will",
"who're":"who are",
"who's":"who is",
"who've":"who have",
"won't":"will not",
"wouldn't":"would not",
"you'd":"you had",
"you'll":"you will",
"you're":"you are",
"you've":"you have",
}

import math
def cosine(vec1, vec2):
    sim = 0
    v2 = dict(vec2)
    for k,v in vec1:
        sim += v * v2.get(k, 0)
    norm1 = math.sqrt(sum([v*v for tid, v in vec1]))
    norm2 = math.sqrt(sum([v*v for tid, v in vec2]))
    denom = (norm1 * norm2)
    if denom < .000000001:
        return 0
    return sim / denom


def clean_url(u):
    return u.split('?')[0]


import re
splits = re.compile(r'[-/,;]')
poss = re.compile(r"'s\b")
bad = re.compile(r'[^0-9a-zA-Z\s]')
compact = re.compile(r'\s+')
sent = re.compile(r'[.!?]')

def normalize(s):
    s = s.lower()
    s = contractions.get(s, s)
    s = poss.sub('', s)
    s = splits.sub(' ', s)
    s = bad.sub('', s)
    return s.strip()

def tokenize(s):
    return [w for w in normalize(s).split() if w not in stop_words and len(w) > 1]

def tokenize_keep_all(s):
    return [w for w in normalize(s).split() if w]


def tokenize_with_sentinels(s):
    s = sent.sub(' SSENTT ', s)
    return [w for w in normalize(s).split() if w]

def bag_of_words(s):
    words = tokenize(s)
    bag = {}
    for word in words:
        if word in bag:
            bag[word] += 1
        else:
            bag[word] = 1
    # n = float(len(words))
    # for k, v in bag.items():
    #     bag[k] = v/n
    return bag



# import MySQLdb
# import sqlite3
# dictionary = {}
# terms = {}

def add_terms(s):
    for t in tokenize(s):
        if len(t) < 3 or t in stop_words:
            continue
        if t in terms:
            terms[t] += 1
        else:
            terms[t] = 1

