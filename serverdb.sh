source vpoc/bin/activate
python serverdb.py
# gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 serverdb:app
