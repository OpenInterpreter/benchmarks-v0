# So I immediately know if I accidentally build this file.
as;oidhas;odifjsaldof

# FROM python:3.11.8
FROM ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest

# Set environment variables you want the worker to have access to here:
# ENV GROQ_API_KEY ...

COPY script-worker/ worker
RUN python -m pip install -r worker/requirements.txt

ENTRYPOINT [ "python", "-m", "worker.run" ]