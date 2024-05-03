import torch
from model.JointBert import JointBERT
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import os
import logging
import numpy as np
from tqdm import tqdm, trange
from JoinrBERT_utils import get_slot_labels, get_intent_labels

logger = logging.getLogger(__name__)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(os.environ.get('SAVE_MODEL_DIR')):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = JointBERT.from_pretrained(os.environ.get('SAVE_MODEL_DIR'),
                                          intent_label_lst=get_intent_labels(),
                                          slot_label_lst=get_slot_labels())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters in the model: {total_params}")
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def convert_input_file_to_tensor_dataset(sentences, tokenizer, pad_token_label_id,
                                         sequence_a_segment_id, cls_token_segment_id,
                                         pad_token_segment_id, mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for sentence in sentences:
        tokens = []
        slot_label_mask = []
        for word in sentence:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        max_len = int(os.environ.get('MAX_LEN'))

        special_tokens_count = 2
        if len(tokens) > max_len - special_tokens_count:
            tokens = tokens[:max_len - special_tokens_count]
            slot_label_mask = slot_label_mask[:max_len - special_tokens_count]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = int(os.environ.get('MAX_LEN')) - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)
    return dataset


def predict(model, tokenizer, sentences, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    intent_label_lst = get_intent_labels()
    slot_label_lst = get_slot_labels()

    pad_token_label_id = int(os.environ.get('IGNORE_INDEX'))
    dataset = convert_input_file_to_tensor_dataset(sentences, tokenizer, pad_token_label_id,
                                                   int(os.environ.get('SEQUENCE_A_SEGMENT_ID')),
                                                   int(os.environ.get('CLS_TOKEN_SEGMENT_ID')),
                                                   int(os.environ.get('PAD_TOKEN_SEGMENT_ID')),
                                                   mask_padding_with_zero=True)

    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=num_samples)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
            outputs = model(**inputs)
            _, (intent_proba, slot_proba), (intent_logits, slot_logits) = outputs[:3]
            proba_results = [[(intent_label_lst[index], proba) for index, proba in enumerate(text)] for text in intent_proba]

            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)
    proba_preds = [proba_results[index][value] for index, value in enumerate(intent_preds)]
    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    intent_preds = [intent_label_lst[intent] for intent in intent_preds]

    return slot_preds_list, intent_preds, proba_preds
