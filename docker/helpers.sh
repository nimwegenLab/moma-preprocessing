# Below are helper functions that are used by the docker script.

function get_version() {
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  cat "${DIR}/../mmpreprocesspy/mmpreprocesspy/VERSION_INFO"
}

function get_image_basename() {
  echo "michaelmell/mmpreprocesspy"
}

function get_singularity_container_name() {
  local CONTAINER_NAME=$(get_image_basename)
  local VERSION=$(get_version)
  local CONTAINER_TAG="${CONTAINER_NAME}_${VERSION}.sif"
  echo "${CONTAINER_TAG}"
}

function get_image_tag() {
  local CONTAINER_NAME=$(get_image_basename)
  local VERSION=$(get_version)
  local CONTAINER_TAG="${CONTAINER_NAME}:${VERSION}"
  echo "${CONTAINER_TAG}"
}

function get_docker_hub_tag() {
  local IMAGE_TAG=$(get_image_tag)
  echo "${DOCKER_USER}/${IMAGE_TAG}"
}
