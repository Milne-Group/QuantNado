FROM r-base:4.5.3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bedtools \
    && rm -rf /var/lib/apt/lists/*
