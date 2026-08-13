ARG BASE_IMAGE=srbench/base
FROM ${BASE_IMAGE} AS build

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ninja-build git && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
ARG ALGORITHM
ENV ALGORITHM="alphasymbolic"
WORKDIR "/install"
COPY --chown=$MAMBA_USER:$MAMBA_USER base_environment.yml scripts/install_algorithm.sh ./
COPY --chown=$MAMBA_USER:$MAMBA_USER ./algorithms/${ALGORITHM} ./${ALGORITHM}
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN bash install_algorithm.sh ${ALGORITHM} && micromamba clean --all --yes

WORKDIR "/srbench"
USER root
