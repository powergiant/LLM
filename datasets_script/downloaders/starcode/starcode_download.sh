# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR=$(realpath $(dirname "$BASH_SOURCE"))
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname $(dirname "$PARENT_DIR"))

CACHE_DIR=$(cat "$PROJECT_DIR"/cache_folder.txt)

DOWNLOAD_DIR="$CACHE_DIR"/step0_rawdata/starcode

if [ ! -d "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
fi
# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
# --include java/*parquet --include cpp/*parquet --include python/*parquet
"$PARENT_DIR"/hfd.sh bigcode/starcoderdata --dataset --tool aria2c -x 16 --local-dir "$DOWNLOAD_DIR" --include *cpp*parquet --hf_token xxx --hf_username xxx
