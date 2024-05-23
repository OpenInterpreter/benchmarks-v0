FROM python:3.11.8

COPY worker/ worker
RUN python -m pip install -r worker/requirements.txt

ENTRYPOINT [ "python", "-m", "worker.run" ]