FROM --platform=amd64 python:3.8-slim-buster

ENV NEO4J_HOST="neo4j://feature-discovery-neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="pass"
ENV NEO4J_DATABASE="alldatamixed"

ENV EXPERIMENT_TYPE="run_all"
ENV TFD_ROOT_FOLDER="/app"

COPY src /app/src
COPY joined-df /app/joined-df
COPY pyproject.toml /app/pyproject.toml
COPY data /app/data
COPY results /app/results

WORKDIR /app

# Install python deps
RUN pip install .

# CMD feature-discovery-cli ${EXPERIMENT_TYPE} --dataset-labels school