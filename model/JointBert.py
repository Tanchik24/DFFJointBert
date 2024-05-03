import os
from dotenv import load_dotenv
from transformers import BertPreTrainedModel, AutoModel
import torch.nn as nn
import torch

load_dotenv()


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = AutoModel.from_pretrained(os.environ.get('DATA_PRETRAINED_PATH'))
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, float(os.environ.get('DROPOUT_RATE')))
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, float(os.environ.get('DROPOUT_RATE')))

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        slot_probs = torch.softmax(slot_logits, dim=-1)

        total_loss = 0

        # 1. Intent Softmax
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=int(os.environ.get('IGNORE_INDEX')))
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += float(os.environ.get('SLOT_LOSS_COEFFICIENT')) * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]
        outputs = ((intent_probs, slot_probs),) + outputs
        outputs = (total_loss,) + outputs
        return outputs
