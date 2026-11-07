FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake ninja-build clang lld git pkg-config \
    libgtk-3-dev python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["cmake", "--version"]
