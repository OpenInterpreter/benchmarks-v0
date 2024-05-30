FROM python:3.11.8

# Set environment variables you want the worker to have access to here:
# ENV GROQ_API_KEY ...

COPY worker/ worker
RUN python -m pip install -r worker/requirements.txt

ENTRYPOINT [ "python", "-m", "worker.run" ]