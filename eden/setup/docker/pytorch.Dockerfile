FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.2-cuda12.1.0

RUN apt-get update
# && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6  # OpenCV dependencies

RUN mkdir -p /app/{datasets,logs,models}

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
# RUN pip install torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# RUN git config --global --add safe.directory /app

COPY .env /app/.env
COPY config /app/config
COPY avr /app/avr

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:."
ENTRYPOINT ["python"]
