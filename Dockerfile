FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN echo "===== requirements.txt ====="
RUN cat requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN pip show python-dotenv

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]