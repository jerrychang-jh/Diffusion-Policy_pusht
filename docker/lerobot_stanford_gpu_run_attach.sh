IMG="lerobot-stanford-pusht:pytorch2.7.1-cuda11.8-cudnn9"

CONTAINER_ID=$(docker ps -aqf "ancestor=${IMG}")

if [ ${CONTAINER_ID} ]; then
  echo "Attach to docker container ${CONTAINER_ID}"
  xhost +
  docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it ${CONTAINER_ID} bash
  xhost -
  return
fi

docker run -it \
    --rm \
    --name lerobot_stanford_container \
    --user "root:root" \
    --gpus all \
    --privileged \
    -v ${PWD}:/workspace \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e DISPLAY=${DISPLAY} \
    -e LINES="$(tput lines)" \
    -w /workspace \
    ${IMG} \
    bash