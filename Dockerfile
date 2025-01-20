FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install -r requirements.txt

COPY ./app.py /app

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

EXPOSE 3000
#CMD [ "fastapi", "run", "app.py", "--port", "3000" ]
# Try to make (internal) redirects be over https
CMD [ "uvicorn", "app:app", "--port", "3000", "--proxy-protocol", "--forwarded-allow-ips", "*" ]
