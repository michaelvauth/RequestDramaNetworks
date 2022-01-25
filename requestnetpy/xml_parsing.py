import xml.etree.ElementTree as ET
import re
import nltk


namespace = '{http://www.tei-c.org/ns/1.0}'
attrib_namespace = '{http://www.w3.org/XML/1998/namespace}'


def get_title(xml_root):
    return list(xml_root.iter(namespace + 'title'))[0].text


def get_year(drama_id, drama_list):
    return int([d['yearNormalized'] for d in drama_list if d['name'] == drama_id][0])


def get_character_ids(xml_root):
    return [
        (
            person.attrib[attrib_namespace + 'id'],
            person.attrib['sex']
        ) for person in xml_root.iter(namespace + 'person')
    ]


def get_scenes(xml_root):
    div_elements = [
        item for item in xml_root.iter(namespace + 'div')
    ]
    div_elements = [item for item in div_elements if 'type' in item.attrib]
    if len(div_elements) < 1:
        print('No scenes or acts found')

    scenes = [
        item for item in div_elements if item.attrib['type'] == 'scene'
    ]
    acts = [
        item for item in div_elements if item.attrib['type'] == 'act'
    ]

    if len(scenes) > 0:
        return scenes
    else:
        return acts


def get_character_list(scene):
    """
    Takes TEI <div type=scene> XML element and returns all speaker ID
    For configuration and communication networks
    """
    return [
        sp.attrib['who'][1:] for sp in scene.iter(namespace + 'sp') if 'who' in sp.attrib
    ]


def get_characters_per_scene(scene_list):
    """
    :param scene_list: List of  TEI <div type=scene> XML elements
    :return: Set of character for each scene in text course order.
    """
    return [
        set(get_character_list(scene)) for scene in scene_list
    ]


def get_character_list_per_scene(scene_list):
    """
    :param scene_list: List of  TEI <div type=scene> XML elements
    :return: List of character for each scene in text course order.
    """
    return [
        get_character_list(scene) for scene in scene_list
    ]


def get_sp_elements_per_scene(scene):
    return [
        sp for sp in scene.iter(namespace + 'sp')
    ]


def get_sp_list_per_scene(scene_list):
    return [
        get_sp_elements_per_scene(scene) for scene in scene_list
    ]


def clean_speaker_item(xml_element) -> str:
    if 'who' in xml_element.attrib:
        speaker_data = xml_element.attrib['who']
        speaker_data = speaker_data.split('#')[1]
        speaker_data = speaker_data.replace(' ', '')
        return speaker_data
    else:
        return False


def text_cleaning(text: str, replacing_regex=re.compile(r"[A-Z].*?\. ")):
    while '\n' in text:
        text = text.replace('\n', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')

    # replace speaker
    text = re.sub(replacing_regex, '', text, count=1)

    return text


def get_speech_text(xml_element, namespace='{http://www.tei-c.org/ns/1.0}'):
    p_text = [element.text for element in xml_element.iter(
        f'{namespace}p') if isinstance(element.text, str)]
    l_text = [element.text for element in xml_element.iter(
        f'{namespace}l') if isinstance(element.text, str)]

    if len(p_text) > 0:
        return text_cleaning(' '.join(p_text))
    elif len(l_text) > 0:
        return text_cleaning(' '.join(l_text))
    else:
        return ''


def speaker_speech_list(tei_file_dir: str, namespace='{http://www.tei-c.org/ns/1.0}') -> list:
    tree = ET.parse(tei_file_dir)
    root = tree.getroot()
    speaker_list = [
        clean_speaker_item(element)
        for element in root.iter(f'{namespace}sp')
        if clean_speaker_item(element)
    ]
    speech_list = [
        get_speech_text(element)
        for element in root.iter(f'{namespace}sp')
        if clean_speaker_item(element)
    ]
    return list(zip(speaker_list, speech_list))


def analyze_sp_text(sp_text, search_token, max_token=50):
    sentences = nltk.sent_tokenize(sp_text, 'german')

    if len(sentences) > 0:
        last_token = nltk.word_tokenize(sentences[-1], 'german')
        if len(last_token) < max_token:     # exclude very long character speech
            if set(set(search_token) & set(last_token)):
                return True


if __name__ == '__main__':
    play_dir = '../corpus/eichendorff-der-letzte-held-von-marienburg.xml'
    tree = ET.parse(play_dir)
    root = tree.getroot()
    scenes = get_scenes(root)
    c = get_characters_per_scene(scenes)
    s = get_sp_list_per_scene(scenes)

    print(get_title(root))
    print(get_character_ids(root))
