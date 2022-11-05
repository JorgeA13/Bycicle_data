FROM python:3.10-slim-buster

WORKDIR /project

COPY requirements.txt /project

RUN pip3 install -r requirements.txt

COPY Analisis ./Analisis

COPY Datos ./Datos

COPY model ./model

CMD python3 /project/model/Modelo_entrenamiento.py



