FROM python:3.6-alpine
COPY * /code/
WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]

