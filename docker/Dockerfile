##### BASE IMAGE #####
FROM continuumio/miniconda3:22.11.1

ARG build_dir="build_dir"
RUN mkdir -p $build_dir
COPY ../mmpreprocesspy /$build_dir/mmpreprocesspy
COPY ../conda_package_list.txt /$build_dir/conda_package_list.txt
COPY ../conda_setup.sh /$build_dir/conda_setup.sh

RUN chmod +x /$build_dir/conda_setup.sh
RUN /$build_dir/conda_setup.sh