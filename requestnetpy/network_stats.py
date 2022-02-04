import os
import re
import pandas as pd


def get_plays_by_author(authors: list) -> list:
    metric_files = os.listdir('network_metrics/')
    author_corpus = []
    for author in authors:
        for play in metric_files:
            if play.startswith(f'{author}-'):
                author_corpus.append(play)

    return author_corpus


def load_corpus_metrics(author_filter: list = None) -> pd.DataFrame:
    corpus_df = pd.DataFrame()
    author_regex = re.compile(f'(\A.*?)-')
    for file in os.listdir('network_metrics/'):
        if author_filter:
            if file in author_filter:
                play_df = pd.read_csv(f'network_metrics/{file}')
                play_df.rename({'Unnamed: 0': 'character'},
                               axis='columns', inplace=True)
                play_df.loc[:, 'title'] = [file[:-4]] * len(play_df)
                author = re.match(author_regex, file).group(0)
                play_df.loc[:, 'author'] = [author[:-1]] * len(play_df)
                corpus_df = pd.concat([corpus_df, play_df])
        else:
            play_df = pd.read_csv(f'network_metrics/{file}')
            play_df.rename({'Unnamed: 0': 'character'},
                           axis='columns', inplace=True)
            play_df.loc[:, 'title'] = [file[:-4]] * len(play_df)
            author = re.match(author_regex, file).group(0)
            play_df.loc[:, 'author'] = [author[:-1]] * len(play_df)
            corpus_df = pd.concat([corpus_df, play_df])
    return corpus_df.fillna(0).sort_values(by=['year'])


def remove_author_from_title(title: str) -> str:
    for item in ['lessing', 'schiller', 'kleist']:
        if item in title:
            return title.replace(f'{item}-', '')
