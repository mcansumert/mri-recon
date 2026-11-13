FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir --break-system-packages h5py matplotlib sigpy

WORKDIR /work
ENV PYTHONPATH=/work/src