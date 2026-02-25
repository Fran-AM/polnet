FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# System deps (libgl1 + libxt6 needed by VTK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libxt6 && \
    rm -rf /var/lib/apt/lists/*

# IMOD (required for TEM simulation: xyzproj, tilt, alterheader)
# IMOD 5.x has a bug: the installer script hard-codes imod_/ but the archive
# uses a versioned path (imod_5.x.y/).  We detect the real name from the
# embedded tar and patch the script before running it.
COPY docker/imod_installer.sh /tmp/imod_install.sh
RUN chmod +x /tmp/imod_install.sh && \
    SKIP=$(grep -n '^exit' /tmp/imod_install.sh | tail -1 | cut -d: -f1) && \
    VER_DIR=$(tail -n +$((SKIP + 1)) /tmp/imod_install.sh | \
              tar tzf - 2>/dev/null | grep '/installIMOD$' | head -1 | cut -d/ -f1) && \
    if [ -n "$VER_DIR" ] && [ "$VER_DIR" != "imod_" ]; then \
        sed -i "s|imod_/|${VER_DIR}/|g" /tmp/imod_install.sh; \
    fi && \
    /tmp/imod_install.sh -y -dir /opt/imod && \
    rm /tmp/imod_install.sh

ENV IMOD_DIR=/opt/imod/IMOD
ENV PATH="${IMOD_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${IMOD_DIR}/lib:${LD_LIBRARY_PATH:-}"

# Python package — install core polnet
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy input data and example configs
COPY data/ data/
COPY config/ config/

# Entrypoint: polnet CLI
ENTRYPOINT ["polnet"]
CMD ["/app/config/all_features.yaml"]