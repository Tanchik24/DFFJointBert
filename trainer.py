import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from JoinrBERT_utils import get_slot_labels, get_intent_labels, compute_metrics
from model.JointBert import JointBERT
from DataPreprocessorForJointBert import load_and_cache_examples

logger = logging.getLogger(__name__)
load_dotenv()


class Trainer:
    def __init__(self, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.intent_label_lst = get_intent_labels()
        self.slot_label_lst = get_slot_labels()

        self.config = BertConfig.from_pretrained(os.environ.get('DATA_PRETRAINED_PATH'))
        self.model = JointBERT.from_pretrained(os.environ.get('DATA_PRETRAINED_PATH'), config=self.config,
                                               intent_label_lst=self.intent_label_lst,
                                               slot_label_lst=self.slot_label_lst)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
                                      batch_size=int(os.environ.get('TRAIN_BATCH_SIZE')))

        if int(os.environ.get('MAX_STEPS')) > 0:
            t_total = int(os.environ.get('MAX_STEPS'))
            train_epochs = int(os.environ.get('MAX_STEPS')) // len(train_dataloader) // int(
                os.environ.get('GRADIENT_ACCUMULATION_STEPS')) + 1
        else:
            t_total = len(train_dataloader) // int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) * int(
                os.environ.get('NUM_TRAIN_EPOCHS'))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': float(os.environ.get('WEIGHT_DECAY'))},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(os.environ.get('LEARNING_RATE')),
                          eps=float(os.environ.get('ADAM_EPSILON')))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=float(os.environ.get('WARMUP_STEPS')),
                                                    num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", os.environ.get('NUM_TRAIN_EPOCHS'))
        logger.info("  Total train batch size = %d", os.environ.get('TRAIN_BATCH_SIZE'))
        logger.info("  Gradient Accumulation steps = %d", os.environ.get('GRADIENT_ACCUMULATION_STEPS'))
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", os.environ.get('LOGGING_STEPS'))
        logger.info("  Save steps = %d", os.environ.get('SAVE_STEPS'))

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(os.environ.get('NUM_TRAIN_EPOCHS')), desc="Epoch")
        losses_train = []
        losses_valid = []
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            batch_loss_train = []
            batch_loss_valid = []
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                outputs = self.model(**inputs)
                loss = outputs[0]
                if int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) > 1:
                    loss = loss / int(os.environ.get('GRADIENT_ACCUMULATION_STEPS'))

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % int(os.environ.get('GRADIENT_ACCUMULATION_STEPS')) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(os.environ.get('MAX_GRAD_NORM')))

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if int(os.environ.get('LOGGING_STEPS')) > 0 and global_step % int(os.environ.get('LOGGING_STEPS')) == 0:
                        results = self.evaluate("dev")
                        batch_loss_train.append(loss.item())
                        batch_loss_valid.append(results['loss'])
                        print(results)

                    if int(os.environ.get('SAVE_STEPS')) > 0 and global_step % int(os.environ.get('SAVE_STEPS')) == 0:
                        self.save_model()

                if 0 < int(os.environ.get('MAX_STEPS')) < global_step:
                    epoch_iterator.close()
                    break
            losses_train.append(np.mean(batch_loss_train))
            losses_valid.append(np.mean(batch_loss_valid))
            if 0 < int(os.environ.get('MAX_STEPS')) < global_step:
                train_iterator.close()
                break
        return global_step, tr_loss / global_step, losses_train, losses_valid

    def evaluate(self, mode):
        dataset = self.dev_dataset

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=int(os.environ.get('EVAL_BATCH_SIZE')))

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", int(os.environ.get('EVAL_BATCH_SIZE')))

        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}

                outputs = self.model(**inputs)
                tmp_eval_loss, _, (intent_logits, slot_logits) = outputs[:3]
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        intent_preds = np.argmax(intent_preds, axis=1)

        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != os.environ.get('PAD_TOKEN_LABEL_ID'):
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        if not os.path.exists(os.environ.get('SAVE_MODEL_DIR')):
            os.makedirs(os.environ.get('SAVE_MODEL_DIR'))
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(os.environ.get('SAVE_MODEL_DIR'))

        logger.info("Saving model checkpoint to %s", os.environ.get('SAVE_MODEL_DIR'))

    def load_model(self):
        if not os.path.exists(os.environ.get('SAVE_MODEL_DIR')):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = JointBERT.from_pretrained(os.environ.get('SAVE_MODEL_DIR'),
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


def plot_losses(losses_train, losses_valid):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_train, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses_valid, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


model_dir = os.environ.get('DATA_PRETRAINED_PATH')
tokenizer = BertTokenizer.from_pretrained(model_dir)
train_dataset = load_and_cache_examples(tokenizer, 'train')
dev_dataset = load_and_cache_examples(tokenizer, 'dev')

trainer = Trainer(train_dataset, dev_dataset)
global_step, loss, losses_train, losses_valid = trainer.train()
plot_losses(losses_train, losses_valid)