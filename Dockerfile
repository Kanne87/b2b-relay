# syntax=docker/dockerfile:1
# b2b-relay Live-Server (Scheibe 2a): jackd (Dummy-Treiber) + JackTrip-Hub-Server + b2b-relay (JACK-Client, API).
#
# Ziele:  runtime (Standard, Serverbetrieb)  |  testclient (runtime + tc/iproute2 + Test-DJ, nur Integrationstest)
#
# JackTrip wird aus dem Quelltext OHNE GUI gebaut (nogui, ohne Virtual Studio). Lizenz: Kern MIT (Expat),
# das Classic-GUI ist GPL-3 und wird hier nicht gebaut.
ARG UBUNTU=ubuntu:24.04

FROM ${UBUNTU} AS jacktrip-build
ARG JACKTRIP_VERSION=v2.7.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git meson ninja-build pkg-config \
      qt6-base-dev qmake6 libjack-jackd2-dev python3-yaml python3-jinja2 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${JACKTRIP_VERSION}" https://github.com/jacktrip/jacktrip.git /src/jacktrip
WORKDIR /src/jacktrip
RUN meson setup builddir --buildtype=release -Dnogui=true -Dnovs=true -Dnoupdater=true \
      -Drtaudio=disabled -Dlibsamplerate=disabled -Dqtversion=6 \
    && ninja -C builddir \
    && install -D builddir/jacktrip /out/usr/local/bin/jacktrip

FROM ${UBUNTU} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      jackd2 libjack-jackd2-0 libqt6core6t64 libqt6network6t64 \
      python3 python3-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=jacktrip-build /out/usr/local/bin/jacktrip /usr/local/bin/jacktrip
RUN python3 -m venv /opt/venv
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[live]"
COPY docker/entrypoint.sh /usr/local/bin/b2b-entrypoint
RUN useradd --system --create-home --uid 10001 b2b && chmod +x /usr/local/bin/b2b-entrypoint
USER 10001
ENV B2B_API_PORT=8090 \
    JACK_RATE=48000 \
    JACK_PERIOD=256
# TCP 4464: JackTrip-Hub-Handshake; UDP 61002-62000: Hub-Worker (je Client ein Port); TCP 8090: HTTP-API
EXPOSE 4464/tcp 8090/tcp 61002-62000/udp
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz'%os.environ.get('B2B_API_PORT','8090'),timeout=3)"]
ENTRYPOINT ["/usr/local/bin/b2b-entrypoint"]

FROM runtime AS testclient
# hadolint ignore=DL3002
USER 0
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*
COPY tests/integration/client.py tests/integration/signals.py /opt/itest/
COPY docker/client-entrypoint.sh /usr/local/bin/client-entrypoint
RUN chmod +x /usr/local/bin/client-entrypoint
# tc braucht NET_ADMIN (cap_add im Compose) und damit root im Container
ENTRYPOINT ["/usr/local/bin/client-entrypoint"]
HEALTHCHECK NONE
