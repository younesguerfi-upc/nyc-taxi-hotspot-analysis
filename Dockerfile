# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY model/ /app/model/
COPY app2.py /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app2.py", "--server.port=8501", "--server.address=0.0.0.0"]