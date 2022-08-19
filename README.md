`MMDeployでJetson AGX Orinの物体検出速度をAGX Xavierと比較してみた` の実験コードです。

# How to use
1. `configs/*` のコンフィグファイルを編集して使用するモデルを選択する
    - 記法は `space/mmdeploy/tools/regression_test.py` や `space/mmdeploy/tests/regression/*.yml` を参照
    - `mmdet` 以外のコードベースを使いたい場合は `regression_test.sh` で指定しているオプション `--codebase` の変更が必要
2. `docker build -t mmdeploy-jetson .`
3. `./regression_test.sh mmdeploy-jetson <path to COCO dataset>`
    - `mmdeploy_regression_working_dir` に変換モデルと実験結果(xlsxファイル)が入る
    - `mmdeploy_checkpoints` にダウンロードされた学習済みモデルがキャッシュされる
4. `./show_results.sh mmdeploy-jetson` でターミナル上に実験結果を表示