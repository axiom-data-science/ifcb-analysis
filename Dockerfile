# NOTE: nvidia drivers and container-toolkit must be installed on the Docker host
# *all* other dependencies are installed in the container using pip
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
FROM python:3.9

RUN useradd --create-home --home-dir=/srv/ifcb-analysis ifcb

WORKDIR /srv/ifcb-analysis

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY fix-tensorrt-libs.sh .

RUN ./fix-tensorrt-libs.sh

COPY . .

RUN pip install .

USER ifcb

ENTRYPOINT ["process-bins"]
