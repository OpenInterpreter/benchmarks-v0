FROM python:3.11.8

# Set environment variables you want the worker to have access to here:
# ENV GROQ_API_KEY ...

COPY worker/ home/main/worker

RUN python -m pip install -r home/main/worker/requirements.txt
# ENTRYPOINT [ "python", "-m", "worker.run" ]
