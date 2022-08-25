# WecloudData
## MLOps Training
### End-to-end model training and deployment using PyTorch Lightning, Fast API and Docker
#### Multi-label text classification
##### Prepared by WecloudData

In this project, we will train and deploy a PyTorch machine learning model to production environment using FastAPI and Docker. The goal is to serve a trained model as API endpoints inside a Docker container with CUDA support, on a local and remote cloud linux machine. 

This tutorial aims to prepare Data Scientists or Machine Learning Engineers for ML Model deployment.

Topics cover in this article:

- PyTorch training
- FastAPI & pydantic
- uvicorn & gunicorn
- Error Handling and Logging
- Writing pytest test cases
- Building and deploying Docker Image with CUDA

Folder Structure
```
.
├── Dockerfile
├── README.md
├── __pycache__
│   └── serve.cpython-37.pyc
├── checkpoints
│   └── best-checkpoint.ckpt
├── data
│   ├── bn_data.csv
│   ├── dataset.csv
│   ├── dsweekly_data.csv
│   ├── dsweekly_data.json
│   ├── dump
│   │   └── benjel_news
│   │       ├── comments.bson
│   │       ├── comments.metadata.json
│   │       ├── notifications.bson
│   │       ├── notifications.metadata.json
│   │       ├── stories.bson
│   │       ├── stories.json
│   │       ├── stories.metadata.json
│   │       ├── users.bson
│   │       └── users.metadata.json
│   ├── hn_data.csv
│   └── hn_data.json
├── get_data
│   └── scraper.py
├── lightning_logs
│   ├── version_0
│   │   ├── events.out.tfevents.1661395807.CAN-JLC-KDML7H.97858.0
│   │   └── hparams.yaml
│   ├── version_1
│   │   ├── events.out.tfevents.1661396011.CAN-JLC-KDML7H.97858.1
│   │   └── hparams.yaml
│   ├── version_2
│   │   ├── events.out.tfevents.1661396235.CAN-JLC-KDML7H.97858.2
│   │   └── hparams.yaml
│   └── version_3
│       ├── events.out.tfevents.1661396495.CAN-JLC-KDML7H.97858.3
│       └── hparams.yaml
├── model
│   └── model_1.pt
├── preprocess
│   └── create_dataset.ipynb
├── requirements.txt
├── scrape
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   └── scrape_methods.cpython-37.pyc
│   └── scrape_methods.py
├── scripts
│   ├── test_api.sh
│   └── turn_on_endpoint.sh
├── serve
│   ├── __pycache__
│   │   └── serve.cpython-37.pyc
│   └── serve.py
├── test.py
└── training
    ├── checkpoints
    │   └── best-checkpoint.ckpt
    ├── lightning_logs
    │   └── version_0
    │       ├── events.out.tfevents.1661453313.CAN-JLC-KDML7H.25324.0
    │       └── hparams.yaml
    └── training.ipynb
```
- serve/ hosts the FastAPI and inference Python script
- model/ consists of the PyTorch model parameters and any preprocessing module joblib
- preprocess/ consists of data preprocessing Python scripts
- training/ contains the example PyTorch model for this project
- scripts/ contains Bash scripts to turn on endpoints and test api locally

To do:
- documentation
- upload lectures to the right place

References
- https://medium.com/@mingc.me/deploying-pytorch-model-to-production-with-fastapi-in-cuda-supported-docker-c161cca68bb8
- https://github.com/kunal-bhadra/Multilabel-Text-Classification-BERT/blob/master/MultiLabel_Text_Classification.ipynb
- https://fastapi.tiangolo.com/deployment/docker/