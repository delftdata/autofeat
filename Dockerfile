FROM python:3.8-slim-buster

#WORKDIR /feature-discovery

RUN apt-get update --fix-missing
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y default-jre

# Install python deps
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/feature-discovery"