FROM mambaorg/micromamba:1.5.8

USER root

RUN apt-get update && apt-get install -y git procps \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /srv/ifcb-analysis && chown $MAMBA_USER:$MAMBA_USER /srv/ifcb-analysis

WORKDIR /srv/ifcb-analysis

USER $MABMA_UESR

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml requirements.txt /tmp/

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . .

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN ./fix-tensorrt-libs.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "./process.py"]
