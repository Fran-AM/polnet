FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# System deps (libgl1 + libxt6 needed by VTK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget libgl1-mesa-glx libxt6 && \
    rm -rf /var/lib/apt/lists/*

# IMOD (required for TEM simulation: xyzproj, tilt, alterheader)
RUN wget -q --no-check-certificate \
        https://bio3d.colorado.edu/imod/AMD64-RHEL5/imod_4.12.49_RHEL7-64_CUDA12.0.sh \
        -O /tmp/imod_install.sh && \
    chmod +x /tmp/imod_install.sh && \
    /tmp/imod_install.sh -y -dir /opt/imod && \
    rm /tmp/imod_install.sh

ENV IMOD_DIR=/opt/imod/IMOD
ENV PATH="${IMOD_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${IMOD_DIR}/lib:${LD_LIBRARY_PATH}"

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