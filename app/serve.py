import os
import sys

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.logger import logger
from transformers import BertTokenizer

from .config import CONFIG
from .model import Model
from .predict import predict
from .schema import *

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(CONFIG['TOKENIZER_PATH'])

    # Initialize the pytorch model
    model = Model()
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "tokenizer": tokenizer,
        "model": model
    }

THRESHOLD = 0.5

@app.get("/ping")
def ping():
    return {"message": "pong!"}

@app.post('/api/v1/predict',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')
    print(body.sentence)
    # prepare input data
    X = body.sentence

    # run model inference
    y = predict(app.package, [X])
    
    results = {
      "pred": y
    }
    print(results)
    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True, debug=True, log_config="log.ini")
