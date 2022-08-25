FROM python:3.7.8

# 
WORKDIR /app

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /app

#
RUN ls -lah /app/*

#
COPY ./start.sh /start.sh
RUN chmod +x /start.sh

#
EXPOSE 80
CMD ["/start.sh"]
