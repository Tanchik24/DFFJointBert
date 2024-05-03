import os
from pydantic import BaseModel

from transformers import BertTokenizer

from fastapi import FastAPI
import uvicorn

from DataPreprocessorForJointBert import clean_message
from JoinrBERT_utils import extract_entities
from predict import predict, load_model


class Message(BaseModel):
    text: str


class Server:
    def __init__(self):
        self.app = FastAPI()
        self.modle = load_model()
        self.tokenizer = BertTokenizer.from_pretrained(os.environ.get('DATA_PRETRAINED_PATH'))
        self.set_routes()

    def set_routes(self):
        @self.app.post("/get_slots_intent")
        async def process_message(message: Message):
            message = clean_message(message.text)
            message = message.split(' ')
            slot_preds_list, intent_preds, proba_preds = predict(model=self.modle, tokenizer=self.tokenizer,
                                                                 sentences=[message], num_samples=1)
            slots = extract_entities(slot_preds_list[0], message)

            return {"intent": proba_preds[0][0],
                    "proba": proba_preds[0][1].item(),
                    "slots": slots}

    def run(self):
        uvicorn.run(self.app, host="localhost", port=8383, log_level="info")


server = Server()
server.run()
