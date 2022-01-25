import re
import nltk
import pandas as pd
import spacy
from typing import Callable
from collections import Counter
from dataclasses import dataclass

nlp = spacy.load("de_core_news_sm")


def filter_for_speech_elements(df: pd.DataFrame) -> pd.DataFrame:
    """Filters pandas DataFrame reprentation of merged CATMA AnnotationCollections.

    Args:
        df (pd.DataFrame): Merged AnnotationCollection

    Returns:
        pd.DataFrame: Only Annotations of GerDraCor <sp> elements or manual question and request elements.
    """
    return df[
        (df.tag == 'sp') |
        (df.tag == 'question') |
        (df.tag == 'request')
    ][['annotator', 'tag', 'annotation', 'prop:who', 'prop:speaker', 'prop:addressee', 'start_point', 'end_point']]


def text_cleaning(text: str, replacing_regex=re.compile(r"[A-Z].*?\. ")) -> str:
    """Removes line breaks, duouble spaces and speaker string.

    Args:
        text (str): The dramatic speech element <sp> in a TEI annotated play.
        replacing_regex ([type], optional): The regex for speaker. Defaults to re.compile(r"[A-Z].*?\. ").

    Returns:
        str: Cleaned text.
    """
    while '\n' in text:
        text = text.replace('\n', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')

    # replace speaker
    text = re.sub(replacing_regex, '', text, count=1)

    return text


class SpeechRequestAnnotation:
    def __init__(self, row) -> None:
        self.annotation = row['annotation']
        self.tag = row['tag']
        self.start_point = row['start_point']
        self.end_point = row['end_point']
        self.speaker = row['prop:speaker'][0]
        self.addressee = row['prop:addressee'][0] if len(
            row['prop:addressee']) > 0 else None


class SpeechItem:
    def __init__(self, row, annotations: pd.DataFrame) -> None:
        self.t = row['annotation']
        self.text = text_cleaning(row['annotation'])
        self.tag = row['tag']
        self.start_point = row['start_point']
        self.end_point = row['end_point']
        self.speaker = row['prop:who'][0][1:]
        self.manual_annotations = [
            SpeechRequestAnnotation(row=r) for _, r
            in annotations.iterrows()
        ]
        self.request = True if len(self.manual_annotations) > 0 else False

    def eval_speech_request(self, classifier: Callable):
        prediction = classifier(self.text)
        if prediction and self.request:
            return 'True Positive'
        elif prediction and not self.request:
            return 'False Positive'
        elif not prediction and not self.request:
            return 'True Negative'
        elif not prediction and self.request:
            return 'False Negative'

    def last_sentences(self, freq: int = 2) -> str:
        sents = nltk.sent_tokenize(
            text=self.text,
            language='german'
        )
        return sents[-freq]

    def qm_freq(self):
        return len(re.findall(r'\?', self.text))

    def token(self):
        token = nltk.word_tokenize(self.text, 'german')
        return token

    def lemma(self):
        lemmatized_text = ''
        doc = nlp(self.text)
        for token in doc:
            lemmatized_text += f' {token.lemma_}'

        return lemmatized_text


def get_speech_item(ac_df: pd.DataFrame):
    for _, row in ac_df.iterrows():
        if row['tag'] == 'sp':
            annotation_df = ac_df[
                (ac_df.start_point >= row.start_point) &
                (ac_df.end_point <= row.end_point) &
                (ac_df.tag != 'sp')
            ]
            yield SpeechItem(row=row, annotations=annotation_df)


def eval_addressee(ac_df: pd.DataFrame):
    speech_items = list(get_speech_item(
        ac_df=filter_for_speech_elements(ac_df)))
    eval_dict = {'same': 0, 'differ': 0}
    for index, item in enumerate(speech_items[:-1]):
        next_speaker = speech_items[index + 1].speaker
        addressees = [a.addressee for a in item.manual_annotations]
        if len(addressees) > 0:
            if next_speaker not in addressees:
                eval_dict['differ'] += 1
            else:
                eval_dict['same'] += 1
    return eval_dict


def baseline_classifier(annotation: str) -> bool:
    import nltk
    sentences = nltk.sent_tokenize(annotation, 'german')
    if len(sentences) > 0:
        last_sentence = sentences[-1]
        last_sentence_tokens = nltk.word_tokenize(last_sentence, 'german')

        speech_request_token = [
            'Sag', 'Sagt', 'Saget',
            'Sprich', 'Spreche', 'Sprecht',
            'Red', 'Rede', 'Redet',
            'Erklär', 'Erkläre', 'Erklärt',
            'Erzähl', 'Erzähle', 'Erzählt',
            # '!'
        ]
        question_token = [
            'Wie', 'Wieso', 'Weshalb', 'Warum', 'Wann',
            'Wer', 'Wem', 'Wessen', 'Wen',
            '?'
        ]

        for token in last_sentence_tokens:
            if token in speech_request_token:
                return True
            elif token in question_token:
                return True
        return False
    else:
        return False


def f1_score(results: dict):
    precision = results['True Positive'] / \
        (results['True Positive'] + results['False Positive'])
    recall = results['True Positive'] / \
        (results['True Positive'] + results['False Negative'])
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
