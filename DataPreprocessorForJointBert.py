import json
import copy
import string
import logging
from dotenv import load_dotenv
import os
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset
from JoinrBERT_utils import get_slot_labels, get_intent_labels, create_data_path_name, check_if_path_exist

logger = logging.getLogger(__name__)
load_dotenv()


class InputExample:
    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor:
    def __init__(self):
        self.intent_labels = get_intent_labels()
        self.slot_labels = get_slot_labels()

        self.input_text_file = 'seq_in.txt'
        self.intent_label_file = 'label.txt'
        self.slot_labels_file = 'seq_out.txt'

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = f'{set_type}_{i}'
            words = text.split()
            intent_label = self.intent_labels.index(
                intent.strip().replace('\n', '')) if intent.strip().replace('\n',
                                                                            '') in self.intent_labels else self.intent_labels.index(
                "UNK")
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s.strip().replace('\n', '')) if s.strip().replace('\n',
                                                                                             '') in self.slot_labels else self.slot_labels.index(
                        "UNK"))
            print(i)
            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        data_path = os.path.join(os.environ.get('DATA_DIR'), mode)
        logger.info(f"LOOKING AT {data_path}")
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


def convert_examples_to_features(tokenizer, examples, pad_token_label_id,
                                 max_seq_len, sequence_a_segment_id, cls_token_segment_id,
                                 pad_token_segment_id, mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)
            slot_label_id = [int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1)
            slot_labels_ids.extend(slot_label_id)

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(tokenizer, mode):
    processor = JointProcessor()

    cached_features_file = create_data_path_name(mode)

    if check_if_path_exist(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", os.environ.get('PREPROCESSED_PATH'), )
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(tokenizer, examples, int(os.environ.get('PAD_TOKEN_LABEL_ID')),
                                                int(os.environ.get('MAX_LEN')),
                                                int(os.environ.get('SEQUENCE_A_SEGMENT_ID')),
                                                int(os.environ.get('CLS_TOKEN_SEGMENT_ID')),
                                                int(os.environ.get('PAD_TOKEN_SEGMENT_ID')))
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    return dataset


def clean_message(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation)).lower()


model_dir = os.environ.get('DATA_PRETRAINED_PATH')
tokenizer = BertTokenizer.from_pretrained(model_dir)
dataset = load_and_cache_examples(tokenizer, 'train')
dataset = load_and_cache_examples(tokenizer, 'dev')
print(dataset.__dict__['tensors'][4])
