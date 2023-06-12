# Below are helper functions that are used by the docker script.

function get_version() {
  cat ../mmpreprocesspy/mmpreprocesspy/VERSION_INFO
}

function get_image_tag() {
  local VERSION=$(get_version)
  local CONTAINER_TAG="mmpreprocesspy:${VERSION}"
  echo "$CONTAINER_TAG"
}

function get_docker_hub_tag() {
  local IMAGE_TAG=$(get_image_tag)
  echo "$DOCKER_USER/$IMAGE_TAG"
}