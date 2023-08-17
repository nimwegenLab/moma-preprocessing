#!/usr/bin/env bash

: '
This script sets up the preprocessing container and its supporting scripts.
It will use Singularity containers, when Singularity is present. It will try to use Docker otherwise.
'

VERSION=$1
ALLOWED_VERSIONS=("v0.2.0")

if [[ ! " ${ALLOWED_VERSIONS[*]} " =~ " ${VERSION} " ]];
then
  printf "ERROR: Invalid version: %s" "${VERSION}"
  exit 1
fi

function get_moma_bin_directory() {
  echo "$HOME/.moma/$VERSION"
}

function get_image_basename() {
  echo "nimwegenlab/mmpreprocesspy"
}

function get_singularity_container_name() {
  local CONTAINER_NAME=$(get_image_basename)
  local VERSION=$(get_version)
  local CONTAINER_TAG="${CONTAINER_NAME}_${VERSION}.sif"
  echo "${CONTAINER_TAG}"
}

function get_image_tag() {
  local CONTAINER_NAME=$(get_image_basename)
  local CONTAINER_TAG="${CONTAINER_NAME}:${VERSION}"
  echo "${CONTAINER_TAG}"
}

function setup_docker() {
  HOST_SCRIPT_DIR="/host_scripts"
  docker_image_name=$(get_image_tag)
  MOMA_BIN_DIRECTORY=$(get_moma_bin_directory)
  mkdir -p "${MOMA_BIN_DIRECTORY}"
  echo "docker name: ${docker_image_name}"
  docker pull "${docker_image_name}"
  id=$(docker create "${docker_image_name}")
  docker cp "${id}":"${HOST_SCRIPT_DIR}"/mm_dispatch_preprocessing.sh "${MOMA_BIN_DIRECTORY}"
  docker cp "${id}":"${HOST_SCRIPT_DIR}"/moma_preprocess "${MOMA_BIN_DIRECTORY}"
}

function setup_singularity() {
    MOMA_SINGULARITY_CONTAINER_DIR="$HOME/.moma/singularity_images"
#    echo "${MOMA_SINGULARITY_CONTAINER_DIR}"
    echo "$(get_singularity_container_name)"
    SINGULARITY_CONTAINER_PATH="${MOMA_SINGULARITY_CONTAINER_DIR}/$(get_singularity_container_name)"
    echo "${SINGULARITY_CONTAINER_PATH}"
    mkdir "${MOMA_SINGULARITY_CONTAINER_DIR}"
    singularity pull --dir="${MOMA_SINGULARITY_CONTAINER_DIR}" "docker://${docker_image_name}"
}

function main(){
  if command -v docker &> /dev/null
  then
    printf "Using Docker.\n"
    setup_docker
  elif command -v singularity &> /dev/null
  then
    printf "Using Singularity.\n"
    setup_singularity
  else
    printf "ERROR: Could not find Docker or Singularity."
  fi
}

main
