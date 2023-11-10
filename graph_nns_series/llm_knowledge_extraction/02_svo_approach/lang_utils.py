import spacy
import textacy
import re
import matplotlib.pylab as plt
import networkx as nx
import torch as th
import torch_geometric as tg
import torch_geometric.nn as tgnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold

from tqdm.auto import tqdm
from thefuzz import fuzz  
from thefuzz import process  
from collections import defaultdict
from typing import Set, List, Tuple
from spacy.lang.en import English

def get_nlp() -> English:
    # Load spacy model
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe("merge_entities")

    return nlp


def extract_person_entities(spacy_doc: spacy.tokens.doc.Doc) -> Set[str]:
    """Extract person entities from a text. Assumes that person-related
    entities start with the capital letter. Clean entity text from
    special characters and numbers.
    """
    unique_entities = set()
    for ent in spacy_doc.ents:
        if ent.label_ == "PERSON":
            ent_clean = "".join([c for c in ent.text if c.isalpha() or c.isspace()]).strip()
            ent_clean = re.sub('[!,*)@#%(&$_?.^]', '', ent_clean).replace("\n", " ").strip()
            # if ent_clean starts with capital letter        
            if ent_clean and ent_clean[0].isupper():
                unique_entities.add(ent_clean)
    return unique_entities


def merge_similar_entities(entities: Set[str], threshold=80) -> List[str]:
    """ Merge similar entities using fuzzy matching.
    """
    merged_entities = []  
    for entity in entities:
        found = False  
        for idx, merged_entity in enumerate(merged_entities):  
            if fuzz.token_set_ratio(entity, merged_entity) >= threshold:  
                merged_entities[idx] = process.extractOne(entity, [entity, merged_entity])[0]  
                found = True  
                break  
        if not found:  
            merged_entities.append(entity)  
    return merged_entities  


def is_entity_present(entity_set: Set[str], token: spacy.tokens.token.Token, threshold = 80) -> Tuple[bool, str]:
    """Check if a token is present in a set of entities. If yes, return the entity.
    For matching entites and tokens, use fuzzy matching with a threshold.
    """
    for entity in entity_set:  
        if fuzz.partial_ratio(entity, token.text) >= threshold:  
            return True, entity  
    return False, ''

def extract_triplets_for_entities(
        spacy_doc: spacy.tokens.doc.Doc,
        entities: Set[str],
        nlp: English,
        threshold=80):
    """Extracts subject-verb-object triplets from a text only if the subject or object
    is present in the set of entities. Uses fuzzy matching with a threshold.
    """
    triplets = list(textacy.extract.subject_verb_object_triples(spacy_doc))
    stopwords = nlp.Defaults.stop_words

    triplets_with_ents = []  
    for triplet in triplets:
        for sub in triplet.subject:
            if sub.text in stopwords:
                continue
            ent_present_sub, ent_sub = is_entity_present(entities, sub, threshold)
            for obj in triplet.object:
                if obj.text in stopwords:
                    continue
                ent_present_ob, ent_ob = is_entity_present(entities, obj, threshold)
                if ent_present_sub or ent_present_ob:
                    triplets_with_ents.append(
                        (ent_sub if ent_present_sub else sub.text.replace("\n", " ").strip(), 
                         " ".join([tok.text for tok in triplet.verb]).strip(),
                         ent_ob if ent_present_ob else obj.text.replace("\n", " ").strip()))
    return triplets_with_ents 

