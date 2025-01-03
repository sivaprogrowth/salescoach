source vpoc/bin/activate
gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 serverdb:app
