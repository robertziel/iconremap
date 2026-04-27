FROM debian:stable-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 python3-numpy python3-xarray python3-netcdf4 python3-scipy \
        cdo libeccodes-tools nco \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/iconremap
COPY iconremap/ /opt/iconremap/iconremap/
ENV PYTHONPATH=/opt/iconremap

ENTRYPOINT ["python3", "-m", "iconremap"]
