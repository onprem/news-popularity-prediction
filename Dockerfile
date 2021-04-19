FROM python:3.9-slim

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

RUN python -m nltk.downloader stopwords punkt

COPY . .

EXPOSE 5000
CMD /usr/local/bin/gunicorn -w 2 -b :5000 server:app
