FROM python:3.8-buster

COPY api api
COPY tensorflow_taxifare_deep tensorflow_taxifare_deep
COPY my_model.h5 my_model.h5
COPY my_pipeline.joblib my_pipeline.joblib
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.app:app --host 0.0.0.0 --port $PORT
