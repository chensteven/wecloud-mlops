FROM python:3.7.8

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./model /code/model

# 
COPY ./app /code/app

#
COPY ./start.sh /code/start.sh

#
CMD ["bash", "start.sh"]
