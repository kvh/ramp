import pandas
from ramp import *
from ramp.estimators.sk import BinaryProbabilities
import sklearn
from sklearn import naive_bayes
import gensim
import tempfile

try:
    training_data = pandas.read_csv('train.csv')
except IOError:
    raise IOError("You need to download the 'Detecting Insults' dataset \
                  from Kaggle to run this example. \
                  http://www.kaggle.com/c/detecting-insults-in-social-commentary")


tmpdir = tempfile.mkdtemp()
context = DataContext(
              store=tmpdir,
              data=training_data)


base_config = Configuration(
    target='Insult',
    metrics=[metrics.AUC()],
    )

base_features = [
    Length('Comment'),
    Log(Length('Comment') + 1)
]

factory = ConfigFactory(
    base_config,
    features=[
        # first feature set is basic attributes
        base_features,

        # second feature set adds word features
        base_features + [
            text.NgramCounts(
                text.Tokenizer('Comment'),
                mindocs=5,
                bool_=True)],

        # third feature set creates character 5-grams
        # and then selects the top 1000 most informative
        base_features + [
            trained.FeatureSelector(
                [text.NgramCounts(
                    text.CharGrams('Comment', chars=5),
                    bool_=True,
                    mindocs=30)
                ],
                selector=selectors.BinaryFeatureSelector(),
                n_keep=1000,
                target=F('Insult')),
            ],

        # the fourth feature set creates 100 latent vectors
        # from the character 5-grams
        base_features + [
            text.LSI(
                text.CharGrams('Comment', chars=5),
                mindocs=30,
                num_topics=100),
            ]
    ],

    # we'll try two estimators (and wrap them so
    # we get class probabilities as output):
    model=[
        BinaryProbabilities(
            sklearn.linear_model.LogisticRegression()),
        BinaryProbabilities(
            naive_bayes.GaussianNB())
    ]
)


for config in factory:
    models.cv(config, context, folds=5, repeat=2,
              print_results=True)


def probability_of_insult(config, ctx, txt):
    # create a unique index for this text
    idx = int(md5(txt).hexdigest()[:10], 16)

    # add the new comment to our DataFrame
    d = DataFrame(
            {'Comment':[txt]},
            index=pandas.Index([idx]))
    ctx.data = ctx.data.append(d)

    # Specify which instances to predict with predict_index
    # and make the prediction
    pred, predict_x, predict_y = models.predict(
            config,
            ctx,
            predict_index=pandas.Index([idx]))

    return pred[idx]

