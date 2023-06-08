##### BASE IMAGE #####
FROM continuumio/miniconda3:22.11.1

ARG build_dir="/build_dir"
ARG exec_dir="/preprocessing"

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

#COPY docker/containerized_mm_dispatch_preprocessing.sh $exec_dir/mm_dispatch_preprocessing.sh
#COPY docker/containerized_test_slurm_mm_dispatch_script.sh $exec_dir/test_slurm_mm_dispatch_script.sh

#COPY conda_setup.sh $build_dir/conda_setup.sh
#RUN chmod +x $build_dir/conda_setup.sh
#RUN $build_dir/conda_setup.sh

ENTRYPOINT ["/preprocessing/call_preproc_fun.sh"]
#ENTRYPOINT ["echo", "hell world"]
