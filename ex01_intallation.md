# 1. Installation
## 1.1 Versions
- Ubuntu 18.04 LTS
- Docker 20.10.22 (> 19.03)
## 1.2 Installation NGC
### Setting up Docker([Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-docker))
Docker-CE on Ubuntu can be setup using Docker’s official convenience script:
```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

Setup the package repository and the GPG key:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Install the nvidia-docker2 package (and dependencies) after updating the package listing:
```bash
sudo apt-get update
```
```bash
sudo apt-get install -y nvidia-docker2
```
Restart the Docker daemon to complete the installation after setting the default runtime:
```bash
sudo systemctl restart docker
```


## 1.3 Installation nvidia-container
**Note** If Docker is updated to 19.03 on a system which already has `nvidia-docker` or `nvidia-docker2` insstalled, then the corresponging methods can still be used.  
To use the native support on a new installation of Docker, first enable the new GPU support in Docker.
```bash
sudo apt-get install -y docker nvidia-container-toolkit
```
## 1.4 Pulling the docker image
```bash
docker pull nvcr.io/nvidia/pytorch:21.12-py3
```
## 1.5 Making docker container
```bash
# Replace 'ssellu`to username on your system.
docker run -it --gpus "device=0" -v /media/hdd/ssellu:/ssellu --name "ssellu_torch" nvcr.io/nvidia/pythorch:21.12-py3 /bin/bash
```
