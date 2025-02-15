FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip uninstall -y numpy scikit-learn && pip install numpy==1.23.5 scikit-learn==1.2.2

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]