if [ $# != 1 ]; then
	echo "Usage: $0 <Docker image>"
	exit 1
fi

DOCKERIMG=$1

docker run -it --rm --gpus all \
	-v $PWD/mmdeploy_regression_working_dir:/root/space/mmdeploy_regression_working_dir:ro \
	-w /root/space/mmdeploy_regression_working_dir \
	$DOCKERIMG \
	python3 -c "import pandas as pd;print(pd.read_excel('mmdet_report.xlsx', index_col=0).to_markdown())"
	#python3 -c "import pandas as pd;print(pd.read_excel('mmdet_report.xlsx', index_col=0)[['Model', 'Backend', 'Precision Type', 'FPS', 'box AP']].to_markdown())"
