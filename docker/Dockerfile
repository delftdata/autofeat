FROM --platform=amd64 python:3.8-slim-buster

# Install java for valentine
RUN apt update && apt install -y default-jre htop

ENV NEO4J_HOST="neo4j://feature-discovery-neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="pass"
ENV NEO4J_DATABASE="benchmark"

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY src /src
COPY pyproject.toml /pyproject.toml

# Install python deps
RUN pip install -e .
