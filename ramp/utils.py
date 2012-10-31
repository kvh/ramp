import re
import numpy as np
import random
from hashlib import md5


def _pprint(params):
    """prints object state in stable manner"""
    params_list = list()
    line_sep = ','
    for i, (k, v) in enumerate(sorted(params.iteritems())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%.10f' % (k, v)
        else:
            # use repr of the rest
            this_repr = '%s=%r' % (k, v)
        params_list.append(this_repr)

    lines = ','.join(params_list)
    return lines


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
    return md5(get_np_hashable(obj)).hexdigest()


def get_np_hashable(obj):
    try:
        return np.getbuffer(obj)
    except TypeError:
        return np.getbuffer(obj.flatten())


def get_single_column(df):
    assert len(df.columns) == 1
    return df[df.columns[0]]


re_object_repr = re.compile(r'\sat\s\w+>')

def stable_repr(obj):
    state = _pprint(obj.__getstate__())
    # HACK: replace 'repr's that contain object id references
    state = re_object_repr.sub('>', state)
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


def add_terms(s):
    for t in tokenize(s):
        if len(t) < 3 or t in stop_words:
            continue
        if t in terms:
            terms[t] += 1
        else:
            terms[t] = 1

