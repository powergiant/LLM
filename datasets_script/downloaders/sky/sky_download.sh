# https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR=$(realpath $(dirname "$BASH_SOURCE"))
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname $(dirname "$PARENT_DIR"))

CACHE_DIR=$(cat "$PROJECT_DIR"/cache_folder.txt)

DOWNLOAD_DIR="$CACHE_DIR"/step0_rawdata/sky

if [ ! -d "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
fi
# 方法1：
# 断了没法继续下载
# huggingface-cli download Skywork/SkyPile-150B --repo-type dataset --resume-download --local-dir DOWNLOAD_DIR  --local-dir-use-symlinks False
# 方法2
# 需要先安装aria2c和git-lfs
# https://hf-mirror.com/往下拉有下载教程
"$PARENT_DIR"/hfd.sh Skywork/SkyPile-150B --dataset --tool aria2c -x 16 --local-dir "$DOWNLOAD_DIR"