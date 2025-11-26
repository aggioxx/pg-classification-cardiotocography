FROM python:3.12
LABEL authors="guilhermeaggio"
WORKDIR /model

COPY ./requirements.txt /model/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /model/requirements.txt

COPY ./app /model/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
