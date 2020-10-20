FROM python:3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 38000
CMD ["python", "manage.py", "runserver", "0.0.0.0:38000"]
