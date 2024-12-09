# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR=$(realpath $(dirname "$BASH_SOURCE"))
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname $(dirname "$PARENT_DIR"))

CACHE_DIR=$(cat "$PROJECT_DIR"/cache_folder.txt)


DOWNLOAD_DIR_3_5M="$CACHE_DIR"/step0_rawdata/BELLE/3_5M
DOWNLOAD_DIR_2M="$CACHE_DIR"/step0_rawdata/BELLE/2M

if [ ! -d "$DOWNLOAD_DIR_3_5M" ]; then
  mkdir -p "$DOWNLOAD_DIR_3_5M"
fi

if [ ! -d "$DOWNLOAD_DIR_2M" ]; then
  mkdir -p "$DOWNLOAD_DIR_2M"
fi

# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
"$PARENT_DIR"/hfd.sh BelleGroup/train_3.5M_CN  --dataset --tool aria2c -x 16 --local-dir "$DOWNLOAD_DIR_3_5M"
"$PARENT_DIR"/hfd.sh BelleGroup/train_2M_CN  --dataset --tool aria2c -x 16 --local-dir "$DOWNLOAD_DIR_2M"
