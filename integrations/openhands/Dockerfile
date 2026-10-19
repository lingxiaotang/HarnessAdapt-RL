# Build context: integrations/openhands. Export the final stage with BuildKit.
FROM debian:bookworm-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/harnessadapt-openhands/python
ENV UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.12.11
RUN uv venv --python 3.12.11 /opt/harnessadapt-openhands/venv
COPY requirements.txt /tmp/openhands-requirements.txt
RUN uv pip install --python /opt/harnessadapt-openhands/venv/bin/python -r /tmp/openhands-requirements.txt
COPY manifest.py /tmp/manifest.py
RUN /opt/harnessadapt-openhands/venv/bin/python /tmp/manifest.py

FROM scratch AS runtime
COPY --from=build /opt/harnessadapt-openhands/ /
