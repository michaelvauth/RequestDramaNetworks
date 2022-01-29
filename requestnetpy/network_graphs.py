import os
import difflib
import pandas as pd
import networkx as nx
from pathy import importlib
import plotly.graph_objects as go
from IPython.display import display
from dataclasses import dataclass
from itertools import permutations
from typing import List


@dataclass
class Edge:
    speaker: str
    addressee: str
    text: str
    weight: int


def format_string_list(str_list: list) -> str:
    out_str = '<br>'
    for s in str_list:
        out_str += s[:50]
        if len(s) > 50:
            out_str += '[...]'
        out_str += '<br>'

    return out_str


def get_config_edges(speech_data: pd.DataFrame) -> nx.Graph:
    edge_dict = {}
    for scene in speech_data.scene.unique():
        scene_data = speech_data[speech_data.scene == scene]
        pairs = permutations(scene_data.speaker.unique(), 2)
        for pair in pairs:
            edge_id = f'{pair[0]}->{pair[1]}'
            if edge_id not in edge_dict:
                edge_dict[edge_id] = {
                    'weight': 1,
                    'text': [str(scene)]
                }
            else:
                edge_dict[edge_id]['weight'] += 1
                edge_dict[edge_id]['text'].append(str(scene))

    for edge in edge_dict:
        speaker, addressee = edge.split('->')
        yield Edge(
            speaker=speaker,
            addressee=addressee,
            text=format_string_list(edge_dict[edge]['text']),
            weight=edge_dict[edge]['weight']
        )


def get_com_edges(speech_data=pd.DataFrame) -> list:
    edge_dict = {}
    for index, row in speech_data[:-1].iterrows():
        speaker = row['speaker']
        addressee = speech_data['speaker'][index + 1]
        edge_id = f'{speaker}->{addressee}'
        if edge_id in edge_dict:
            edge_dict[edge_id]['weight'] += 1
            edge_dict[edge_id]['text'].append(row['speeches'])
        else:
            edge_dict[edge_id] = {
                'weight': 1,
                'text': [row['speeches']]
            }

    for edge in edge_dict:
        speaker, addressee = edge.split('->')
        yield Edge(
            speaker=speaker,
            addressee=addressee,
            text=format_string_list(edge_dict[edge]['text']),
            weight=edge_dict[edge]['weight']
        )


def get_request_edges(speech_data: pd.DataFrame) -> list:
    edge_dict = {}

    for index, row in speech_data[:-1].iterrows():
        if row['request_classification'] == 'True':
            speaker = row['speaker']
            addressee = speech_data['speaker'][index + 1]
            edge_id = f'{speaker}->{addressee}'
            if edge_id in edge_dict:
                edge_dict[edge_id]['weight'] += 1
                edge_dict[edge_id]['text'].append(row['speeches'])
            else:
                edge_dict[edge_id] = {
                    'weight': 1,
                    'text': [row['speeches']]
                }

    for edge in edge_dict:
        speaker, addressee = edge.split('->')
        yield Edge(
            speaker=speaker,
            addressee=addressee,
            text=format_string_list(edge_dict[edge]['text']),
            weight=edge_dict[edge]['weight']
        )


def create_network_from_edges(edges: List[Edge], network_type: str) -> nx.DiGraph:
    if network_type in ['com', 'request']:
        di_graph = nx.DiGraph()
        di_graph.add_weighted_edges_from(
            [
                (edge.speaker, edge.addressee, edge.weight)
                for edge in edges
            ]
        )
        return di_graph
    else:
        di_graph = nx.Graph()
        di_graph.add_weighted_edges_from(
            [
                (edge.speaker, edge.addressee, edge.weight)
                for edge in edges
            ]
        )
        return di_graph


def speaker_point(p1_x, p1_y, p2_x, p2_y, distance: float = 0.1):
    return ((1 - distance) * p1_x + distance * p2_x), ((1 - distance) * p1_y + distance * p2_y)


def speaker_addressee_str(edge) -> str:
    return f"{edge.speaker} → {edge.addressee}"


def request_value(network_stats: pd.DataFrame) -> pd.DataFrame:
    return network_stats['outdegree'] * (network_stats['weighted_outdegree'] / network_stats['speeches'])


class Network:
    def __init__(self, gerdracor_play_id: str, network_layout: callable = nx.drawing.layout.kamada_kawai_layout):
        self.id = gerdracor_play_id
        try:
            self.speech_data = pd.read_json(
                f'gerdracor_request_annotations/{gerdracor_play_id}.json', orient='records')
        except ValueError:
            ids = [item[:-5]
                   for item in os.listdir('gerdracor_request_annotations/')]
            closest_ids = difflib.get_close_matches(
                gerdracor_play_id, ids, cutoff=0.2)
            raise ValueError(
                f'Could not find the play. Did you mean any of these: {closest_ids}')

        self.config_edges = list(
            get_config_edges(speech_data=self.speech_data))
        self.com_edges = list(get_com_edges(speech_data=self.speech_data))
        self.request_edges = list(
            get_request_edges(speech_data=self.speech_data))

        self.config_network = create_network_from_edges(
            edges=self.config_edges, network_type='config')
        self.com_network = create_network_from_edges(
            edges=self.com_edges, network_type='com')
        self.request_network = create_network_from_edges(
            edges=self.request_edges, network_type='request')

        self.network_layout = network_layout
        self.pos = network_layout(self.config_network)

    def stats(
            self,
            network_type: str = 'request',
            gender_metadata: dict = None) -> pd.DataFrame:
        stat_network = self.request_network if network_type == 'request' else self.com_network

        bc = nx.betweenness_centrality(stat_network, normalized=True)
        bc_weighted = nx.betweenness_centrality(
            stat_network, normalized=True, weight='weight')
        pr = nx.pagerank(stat_network)
        pr_weighted = nx.pagerank(stat_network, weight='weight')
        degree = stat_network.degree()
        indegree = stat_network.in_degree()
        weighted_indegree = stat_network.in_degree(weight='weight')
        outdegree = stat_network.out_degree()
        weighted_outdegree = stat_network.out_degree(weight='weight')

        speeches = self.com_network.out_degree(weight='weight')

        network_df = pd.DataFrame(
            {
                'speeches': dict(speeches),
                'degree': dict(degree),
                'indegree': dict(indegree),
                'weighted_indegree': dict(weighted_indegree),
                'outdegree': dict(outdegree),
                'weighted_outdegree': dict(weighted_outdegree),
                'betweenness': dict(bc),
                'betweenness_weighted': dict(bc_weighted),
                'pagerank': dict(pr),
                'pagerank_weighted': dict(pr_weighted),
            }
        )
        network_df.loc[:, 'request_value'] = request_value(network_df)

        if gender_metadata:
            character_gender = [
                gender_metadata[self.id][character]
                if character in gender_metadata[self.id]
                else pd.NA for character in network_df.index
            ]
            network_df.loc[:, 'gender'] = character_gender

        network_df = network_df[network_df.degree > 0]
        return network_df.sort_values(by='request_value', ascending=False)

    def plot(
            self,
            network_type: str = 'request',
            node_size: str = 'request_value',
            node_factor: float = 100.0,
            node_alpha: int = 3,
            specified_layout: bool = False,):

        def norm_col(value: float, values: pd.Series) -> float:
            return value / sum(values)

        if network_type == 'request':
            plot_graph = self.request_network
            plot_edges = self.request_edges
            stats = self.stats(network_type=network_type).fillna(value=0)
            speaker_size = dict(
                stats[node_size].apply(
                    norm_col,
                    args=(stats[node_size],)
                )
            )
            if specified_layout:
                self.pos = self.network_layout(self.request_network)
        elif network_type == 'com':
            plot_graph = self.com_network
            plot_edges = self.com_edges
            stats = self.stats(network_type=network_type).fillna(value=0)
            speaker_size = dict(
                stats[node_size].apply(
                    norm_col,
                    args=(stats[node_size],)
                )
            )
            if specified_layout:
                self.pos = self.network_layout(self.com_network)
        else:
            plot_graph = self.config_network
            plot_edges = self.config_edges
            speaker_size = dict(plot_graph.degree())

        fig = go.Figure()
        legend_groups = []
        edge_weight_sum = sum([edge.weight for edge in plot_edges])
        for edge in plot_edges:
            lg = speaker_addressee_str(edge)
            speaker_coordinates = speaker_point(
                p1_x=self.pos[edge.addressee][0],
                p1_y=self.pos[edge.addressee][1],
                p2_x=self.pos[edge.speaker][0],
                p2_y=self.pos[edge.speaker][1],
                distance=0.03
            )
            addressee_coordinates = speaker_point(
                p1_x=self.pos[edge.addressee][0],
                p1_y=self.pos[edge.addressee][1],
                p2_x=self.pos[edge.speaker][0],
                p2_y=self.pos[edge.speaker][1],
                distance=0.97
            )
            # plot edges
            if network_type == 'config':    # plot undirected network
                fig.add_trace(
                    go.Scatter(
                        x=[speaker_coordinates[0], addressee_coordinates[0]],
                        y=[speaker_coordinates[1], addressee_coordinates[1]],
                        opacity=0.3,
                        line={
                            'width': edge.weight / edge_weight_sum * 100 + 1,
                            'smoothing': 1.3,
                            'color': 'grey'
                        },
                        hoverinfo='skip',
                        name=lg,
                        mode='lines',
                        legendgroup=lg,
                        showlegend=False
                    )
                )
            else:                           # plot directed network
                fig.add_annotation(
                    x=speaker_coordinates[0],  # arrows' head
                    y=speaker_coordinates[1],  # arrows' head
                    ax=addressee_coordinates[0],  # arrows' tail
                    ay=addressee_coordinates[1],  # arrows' tail
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    text='',  # if you want only the arrow
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=0.6,
                    arrowwidth=edge.weight / edge_weight_sum * 100 + 1,
                    arrowcolor='grey',
                    opacity=0.3
                )
            legend_groups.append(lg)

            # plot hovertext per edge
            hover_pos = speaker_point(
                p1_x=self.pos[edge.speaker][0],
                p1_y=self.pos[edge.speaker][1],
                p2_x=self.pos[edge.addressee][0],
                p2_y=self.pos[edge.addressee][1]
            )
            if len(edge.text) > 500:
                edge.text = edge.text[:500] + '<br>[...]'
            fig.add_trace(
                go.Scatter(
                    x=[hover_pos[0]],
                    y=[hover_pos[1]],
                    mode='markers',
                    marker_symbol='cross-thin',
                    name=lg,
                    text=f"<b>{lg}</b>:<br>{edge.text}",
                    marker=dict(
                        color='grey',
                        opacity=0.4,
                        size=edge.weight / edge_weight_sum * 100 + 1
                    ),
                    showlegend=False,
                    hoverinfo='text'
                )
            )

        # plot nodes
        for speaker in speaker_size:
            # plot indegree
            fig.add_trace(
                go.Scatter(
                    x=[self.pos[speaker][0]],
                    y=[self.pos[speaker][1]],
                    opacity=1,
                    marker={
                        'size': speaker_size[speaker] * node_factor + node_alpha,
                        'color': 'black',
                        'opacity': 1
                    },
                    text=speaker,
                    textposition='top center',
                    mode='markers + text',
                    legendgroup=speaker,
                    showlegend=False,
                    hoverinfo='text',
                    textfont_size=speaker_size[speaker] *
                    node_factor + node_alpha
                )
            )

        fig.update_layout(
            xaxis={'ticks': '', 'showticklabels': False, 'showgrid': False},
            yaxis={'ticks': '', 'showticklabels': False, 'showgrid': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600,
            title=f'{network_type.upper()} GRAPH FOR {self.id.upper()}'
        )
        fig.show()

        if network_type in ['com', 'request']:
            display(stats)

        return fig

    def save_network_plot(self, network_type: str) -> None:
        fig = self.plot(network_type=network_type)
        fig.write_html(f'network_html_files/{self.id}_{network_type}.html')


if __name__ == '__main__':
    ntest = Network('kleist-d')
    ntest.plot(network_type='request')
