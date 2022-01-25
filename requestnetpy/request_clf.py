from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy
nlp = spacy.load("de_core_news_sm")


def train_request_classifier(manual_annotations: str):
    # manual annotations
    manual_annotations = pd.read_json(
        'manual_annotations.json', orient='records')

    # get text elements
    request_docs = manual_annotations['text']
    labels = manual_annotations['request']

    training_data, test_data, training_label, test_label = train_test_split(
        request_docs,
        labels,
        test_size=0.33,
        shuffle=True
    )

    # take care of inbalanced training data
    class_weight = compute_class_weight(
        class_weight='balanced',
        classes=['True', 'False'],
        y=training_label
    )

    # Create a pipeline
    clf = make_pipeline(
        TfidfVectorizer(
            token_pattern=r"(?u)\b\w\w+\b|\?",
            analyzer='word',
            # stop_words=set(stopwords.words('german')),
            ngram_range=(1, 1)
        ),
        SGDClassifier(
            class_weight={
                'True': class_weight[0],
                'False': class_weight[1]
            },
            loss='hinge',
            penalty='l2',
            alpha=0.001,
            random_state=100,
            max_iter=3,
            tol=None
        ),
    )

    # Fit the model with training set
    clf.fit(training_data, training_label)

    # Predict labels for the test set
    predicted_label = clf.predict(test_data)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=test_label, y_pred=predicted_label, average='weighted')
    print(precision, recall, fscore)

    return clf
