FROM slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342 AS runtime

ARG PIP_INDEX_URL

# Keep Slime's CUDA/PyTorch/SGLang/Megatron stack intact.  R2E-Gym itself is
# mounted from the workspace and placed on PYTHONPATH; only its runtime
# dependencies and Mini-SWE-Agent are added to the official image.
RUN python -m pip install --no-cache-dir \
      mini-swe-agent==2.4.5 \
      litellm==1.92.0 \
      docker==7.2.0 \
      gym==0.26.2 \
      kubernetes==36.0.3 \
      swebench==3.0.2 \
      diff-parser==1.1 \
      simple-parsing==0.1.8 \
      bashlex==0.18 \
      fire==0.7.1 \
    && python -m pip install --no-cache-dir --no-deps protobuf==7.35.1

# Mini-SWE's DockerEnvironment shells out to the Docker CLI. The Python
# ``docker`` package above is not a substitute; the CLI talks to the host
# daemon through the socket mounted by container.sh.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

ENV MSWEA_SILENT_STARTUP=1 \
    MSWEA_COST_TRACKING=ignore_errors \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="slime-r2e-curriculum" \
      org.opencontainers.image.description="Official Slime plus Mini-SWE-Agent and local-Docker R2E runtime dependencies"

# Optional teacher-enabled image. Explicit versions are required because the
# historical CLI package versions were not recorded with the original image.
FROM node:22.14.0-bookworm-slim AS node
FROM runtime AS teachers
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
ARG CODEX_VERSION
ARG OPENCODE_VERSION
RUN test -n "$CODEX_VERSION" && test -n "$OPENCODE_VERSION" \
    && npm install --global "@openai/codex@$CODEX_VERSION" "opencode-ai@$OPENCODE_VERSION" \
    && codex --version && opencode --version
