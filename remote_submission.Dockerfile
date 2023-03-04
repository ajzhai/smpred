FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

RUN /bin/bash -c ". activate habitat"

# Install project specific packages
RUN /bin/bash -c "apt-get update; apt-get install -y libsm6 libxext6 libxrender-dev; . activate habitat; pip install opencv-python"
RUN /bin/bash -c ". activate habitat; pip install --upgrade cython numpy"
RUN /bin/bash -c ". activate habitat; pip install matplotlib seaborn==0.9.0 scikit-fmm==2019.1.30 scikit-image==0.15.0 imageio==2.6.0 scikit-learn==0.22.2.post1 ifcfg"

# Install pytorch and torch_scatter
# RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN /bin/bash -c ". activate habitat; pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html"

# Install detectron2
# RUN /bin/bash -c ". activate habitat; python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html"
RUN /bin/bash -c ". activate habitat; python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html"

# RUN /bin/bash -c ". activate habitat; pip install torchvision==0.11.0 "


RUN /bin/bash -c ". activate habitat; python -m pip install openmim; mim install mmcv-full==1.6.0"
RUN /bin/bash -c ". activate habitat; pip install gym==0.10.5 "
RUN /bin/bash -c ". activate habitat; python -m pip install transformers"
# Install mmseg
ADD mmsegmentation mmsegmentation
RUN /bin/bash -c ". activate habitat; cd mmsegmentation; pip install -e ."
RUN /bin/bash -c ". activate habitat; python -m pip install open3d"

ARG INCUBATOR_VER=unknown
ADD remote_submission.sh remote_submission.sh
ADD configs/challenge_objectnav2021.local.rgbd.yaml /challenge_objectnav2021.local.rgbd.yaml
ADD configs/challenge_objectnav2021.remote.rgbd.yaml /challenge_objectnav2021.remote.rgbd.yaml
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ADD configs/challenge_objectnav2022noisy.local.rgbd.yaml /challenge_objectnav2022noisy.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote
#ADD Stubborn Stubborn
ENV PYTHONPATH "${PYTHONPATH}:/Stubborn"


ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash remote_submission.sh"]
