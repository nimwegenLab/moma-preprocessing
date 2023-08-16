FROM continuumio/miniconda3:22.11.1

ARG build_dir="/build_dir"
ARG exec_dir="/preprocessing"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN mkdir -p $build_dir

# setup conda environment
COPY conda/conda_package_list.txt $build_dir/conda_package_list.txt
COPY conda/conda_environment_setup.sh $build_dir/conda_environment_setup.sh
RUN chmod +x $build_dir/conda_environment_setup.sh
RUN $build_dir/conda_environment_setup.sh

# setup mmpreprocesspy package
COPY mmpreprocesspy $build_dir/mmpreprocesspy
COPY conda/conda_package_setup.sh $build_dir/conda_package_setup.sh
RUN chmod +x $build_dir/conda_package_setup.sh
RUN $build_dir/conda_package_setup.sh

COPY call_preproc_fun.py $exec_dir/call_preproc_fun.py
COPY docker/call_preproc_fun.sh $exec_dir/call_preproc_fun.sh
RUN chmod +x $exec_dir/call_preproc_fun.sh

ARG host_scripts="/host_scripts"
RUN mkdir $host_scripts
COPY docker/mm_dispatch_preprocessing.sh $host_scripts/mm_dispatch_preprocessing.sh
COPY docker/moma_preprocess $host_scripts/moma_preprocess

ENTRYPOINT ["/preprocessing/call_preproc_fun.sh"]
