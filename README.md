# Network Analysis by Automated Request Annotations
This repository provides a python package which automates the detection of speech requests in German dramas by and uses these annotations for network analysis.

## Ressources

### gerdracor_request_annotations/
Speech request annotations for each `<sp>` element in the TEI encoded plays of [GerDraCor](https://dracor.org/ger).
Each `<sp>` element is represented in this structure:
```
{
    "speaker":"dietrich",
    "speeches":"Neun!",
    "scene":1,
    "request_classification":"False"
}
```
These annotations are created by the `requestnetpy.request_clf` (see below).

### network_metrics/
Network stats based on the request annotations.

### `requestnetpy`
#### `.gold_annotation_preprocessing`
Preprocessing functions to create training data from [CATMA](https://catma.de/) annotations.

#### `.network_graphs`
`Network` class which creates three types of drama networks:
- configuration networks based on the character's copresence
- communcation networks based on every character speech
- request networks based on the request annotations

Additionaly, every network type can be plotted as Plotly graph (see: network_ploty.ipynb) and standard network stats for each networks type are provided.

#### `.request_clf`
A Support Vector Machine classifier pipeline to detect dramatic speeches as requests.

#### `.xml_parsing`
Perprocessing for the TEI encoded plays from GerDraCor.

### character_gender_gerdracor.json
The gender of all characters in the GerDraCor corpus. 

### gerdracor-metadata.csv
Corpus metadata.

### manual_annotations.json
Manual request annotations of four German plays:
- kleist-der-zerbrochene-krug
- kleist-die-familie-schroffenstein
- schlegel-canut
- wedekind-fruehlings-erwachen

### network_plots.ipynb
Jupyte Notebook to illustrate the network types with network graph as interactive plotly plots.

## Prerequisites
networkx==2.6.2\
nltk==3.6.7\
plotly==5.5.0\
pandas==1.4.0\
scikit-learn==1.0.2\
spacy==3.2.1
