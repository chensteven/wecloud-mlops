# WecloudData
## MLOps Training
### End-to-end model training and deployment using PyTorch Lightning, Fast API and Docker
#### Multi-label text classification
##### Prepared by WecloudData

In this project, we will train and deploy a PyTorch machine learning model to production environment using FastAPI and Docker. The goal is to serve a trained model as API endpoints inside a Docker container with CUDA support, on a local and remote cloud linux machine. 

This tutorial aims to prepare Data Scientists or Machine Learning Engineers for ML Model deployment.

Topics cover in this article:

- Text data collection and preprocessing
- PyTorch training
- FastAPI & pydantic
- uvicorn & gunicorn
- Error Handling and Logging
- Writing pytest test cases
- Building and deploying Docker Image locally and remotely with CUDA

Folder Structure
```
|── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── log.ini
│   ├── model.py
│   ├── predict.py
│   ├── schema.py
│   ├── serve.py
│   └── tests
│       ├── __init__.py
│       ├── __pycache__
│       └── test_api.py
├── data
│   ├── dump
│   ├── post_categories
│   │   └── dataset.csv
│   └── scraped_data
├── docker_build.sh
├── docker_run_local.sh
├── model
│   ├── bert_pretrained
│   ├── model_1.pt
│   └── tokenizer
├── requirements.txt
├── start.sh
├── training
│   ├── checkpoints
│   ├── lightning_logs
│   └── training.ipynb
└── utils
    ├── create_post_categories_dataset.ipynb
    ├── scrape
    ├── scraper.py
    └── scripts
        ├── test_api.sh
        └── turn_on_endpoint.sh
```
- *app/* hosts the FastAPI and inference Python script
- *model/* consists of the PyTorch model, tokenizer and pretrains
- *utils/* consists of various Python and Bash scripts for data creation and scraping and endpoint setup and testing
- *training/* contains the example PyTorch model for this project

To do:
- documentation
- upload lectures to the right place

References
- https://medium.com/@mingc.me/deploying-pytorch-model-to-production-with-fastapi-in-cuda-supported-docker-c161cca68bb8
- https://github.com/kunal-bhadra/Multilabel-Text-Classification-BERT/blob/master/MultiLabel_Text_Classification.ipynb
- https://fastapi.tiangolo.com/deployment/docker/