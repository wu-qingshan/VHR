#!/usr/bin/env bash
# 数据准备脚本：MSCOCO、POPE、LLaVA-Bench-In-the-Wild
# 与 README 及 datasets.py 中的路径一致，快速可靠下载并解压到 ./data

set -e
DATA_DIR="${1:-./data}"
cd "$(dirname "$0")/.."
mkdir -p "$DATA_DIR"
DATA_DIR="$(realpath "$DATA_DIR")"

echo "==> 数据目录: $DATA_DIR"

# ---------- 1. MSCOCO ----------
COCO_DIR="$DATA_DIR/coco"
mkdir -p "$COCO_DIR/annotations" "$COCO_DIR/val2014"

# val2014 图像（约 6GB）
VALZIP="$COCO_DIR/val2014.zip"
if [ ! -d "$COCO_DIR/val2014" ] || [ -z "$(ls -A $COCO_DIR/val2014 2>/dev/null)" ]; then
  if [ ! -f "$VALZIP" ]; then
    echo "==> 下载 COCO val2014 图像..."
    wget -c -q --show-progress "http://images.cocodataset.org/zips/val2014.zip" -O "$VALZIP"
  fi
  echo "==> 解压 val2014..."
  unzip -o -q "$VALZIP" -d "$COCO_DIR"
  rm -f "$VALZIP"
else
  echo "==> COCO val2014 已存在，跳过"
fi

# 标注（含 train+val，CHAIR 评估需要）
ANNZIP="$COCO_DIR/annotations_trainval2014.zip"
if [ ! -f "$COCO_DIR/annotations/captions_val2014.json" ] || [ ! -f "$COCO_DIR/annotations/instances_val2014.json" ]; then
  if [ ! -f "$ANNZIP" ]; then
    echo "==> 下载 COCO 标注 (train+val)..."
    wget -c -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -O "$ANNZIP"
  fi
  echo "==> 解压标注..."
  unzip -o -q "$ANNZIP" -d "$COCO_DIR"
  rm -f "$ANNZIP"
else
  echo "==> COCO 标注已存在，跳过"
fi

# ---------- 2. POPE (coco) ----------
POPE_DIR="$DATA_DIR/pope/coco"
mkdir -p "$POPE_DIR"
BASE="https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
for name in popular random adversarial; do
  F="$POPE_DIR/coco_pope_${name}.json"
  if [ ! -f "$F" ]; then
    echo "==> 下载 POPE coco_pope_${name}.json ..."
    wget -q -O "$F" "$BASE/coco_pope_${name}.json"
  fi
done
echo "==> POPE 三个子集已就绪"

# ---------- 3. LLaVA-Bench-In-the-Wild ----------
LLAVA_DIR="$DATA_DIR/llava-bench-in-the-wild"
mkdir -p "$LLAVA_DIR"
if [ ! -f "$LLAVA_DIR/questions.jsonl" ] || [ ! -d "$LLAVA_DIR/images" ] || [ -z "$(ls -A $LLAVA_DIR/images 2>/dev/null)" ]; then
  echo "==> 下载 LLaVA-Bench（questions.jsonl + images）..."
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download liuhaotian/llava-bench-in-the-wild --repo-type dataset --local-dir "$LLAVA_DIR" --local-dir-use-symlinks False
  else
    python3 -c "
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='liuhaotian/llava-bench-in-the-wild', repo_type='dataset', local_dir='$LLAVA_DIR', local_dir_use_symlinks=False)
except ImportError:
    print('请先安装: pip install huggingface_hub')
    raise SystemExit(1)
"
  fi
else
  echo "==> LLaVA-Bench 已存在，跳过"
fi

echo ""
echo "==> 数据准备完成。目录结构："
echo ""
echo "  $DATA_DIR"
echo "  ├── coco"
echo "  │   ├── val2014/"
echo "  │   └── annotations/"
echo "  │         ├── captions_val2014.json"
echo "  │         ├── captions_train2014.json"
echo "  │         ├── instances_val2014.json"
echo "  │         └── instances_train2014.json"
echo "  ├── pope/coco/"
echo "  │   ├── coco_pope_popular.json"
echo "  │   ├── coco_pope_random.json"
echo "  │   └── coco_pope_adversarial.json"
echo "  └── llava-bench-in-the-wild/"
echo "        ├── images/"
echo "        └── questions.jsonl"
echo ""
