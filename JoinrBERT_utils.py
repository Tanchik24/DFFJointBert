from dotenv import load_dotenv
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np

load_dotenv()


def get_intent_labels():
    return [label.strip() for label in
            open(os.path.join(os.environ.get("DATA_DIR"), os.environ.get("INTENT_LABELS_FILE_NAME")), 'r',
                 encoding='utf-8')]


def get_slot_labels():
    return [label.strip() for label in
            open(os.path.join(os.environ.get("DATA_DIR"), os.environ.get("SLOT_LABELS_FILE_NAME")), 'r',
                 encoding='utf-8')]


def check_if_path_exist(path: str) -> bool:
    return True if os.path.exists(path) else False


def create_data_path_name(stage: str) -> str:
    data_preprocessed_path = os.path.join(
        os.environ.get('PREPROCESSED_PATH'),
        f'{stage}_{os.environ.get("MAX_LEN")}')
    return data_preprocessed_path


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


def extract_entities(labels: list, words: list):
    entities = []
    current_entity = []
    current_type = None

    for label, word in zip(labels, words):
        if label.startswith('B-'):
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
            current_entity = [word]
            current_type = label[2:]
        elif label.startswith('I-') and current_entity:
            current_entity.append(word)
        elif label == 'O':
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
                current_entity = []
                current_type = None

    if current_entity:
        entities.append((" ".join(current_entity), current_type))

    return entities
