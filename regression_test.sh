#!/bin/bash
if [ $# != 2 ]; then
	echo "Usage: $0 <Docker image> <Path to COCO dataset>"
	exit 1
fi

DOCKERIMG=$1
COCODIR=$(cd $(dirname $2) && pwd)/$(basename $2)

docker run -it --rm --gpus all \
	-v $PWD/mmdeploy_regression_working_dir:/root/space/mmdeploy_regression_working_dir \
	-v $PWD/mmdeploy_checkpoints:/root/space/mmdeploy_checkpoints \
	-v $PWD/configs:/root/space/mmdeploy/tests/regression:ro \
	-v $COCODIR:/root/space/mmdeploy/data/coco:ro \
	-w /root/space/mmdeploy \
	$DOCKERIMG \
	python3 tools/regression_test.py \
		--codebase mmdet \
		--performance
		#--models "SSD"
		#--backends tensorrt

