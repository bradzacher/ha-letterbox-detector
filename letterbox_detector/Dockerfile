ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-numpy \
        python3-opencv \
        python3-requests \
        python3-websockets \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/letterbox
COPY letterbox.py main.py ./
RUN mkdir -p /etc/services.d/letterbox
COPY run-s6.sh /etc/services.d/letterbox/run
RUN chmod +x /etc/services.d/letterbox/run
