# CDFSOD-benchmark Demo Script

このデモスクリプトは、少数ショット物体検出（Few-shot Object Detection）の実行例を提供します。事前学習済みモデルを使用して画像内の物体を検出し、結果を可視化します。

## 使用方法

### 基本的な使用法

```bash
python ../demo.py --image_dir input --output_dir output
```

### 詳細なオプション

```bash
python ../demo.py \
  --config_file configs/open-vocabulary/lvis/vitl.yaml \
  --rpn_config_file configs/RPN/mask_rcnn_R_50_FPN_1x.yaml \
  --model_path weights/trained/open-vocabulary/lvis/vitl_0069999.pth \
  --image_dir input \
  --output_dir output \
  --category_space demo/ycb_prototypes.pth \
  --device cpu \
  --overlapping_mode True \
  --topk 1 \
  --output_pth False \
  --threshold 0.45
```

### パラメータ説明

- `config_file`: モデル設定ファイルのパス
- `rpn_config_file`: RPNの設定ファイルのパス
- `model_path`: 学習済みモデルのパス
- `image_dir`: 入力画像のディレクトリ
- `output_dir`: 出力画像を保存するディレクトリ
- `category_space`: カテゴリプロトタイプファイルのパス
- `device`: 推論に使用するデバイス（'cpu'または'cuda'）
- `overlapping_mode`: 重複した検出を処理するモード
- `topk`: 保持する上位k個の予測
- `output_pth`: 生の予測結果を.pthファイルとして保存するかどうか
- `threshold`: 検出の信頼度閾値

## 入出力形式

### 入力

- 入力ディレクトリ内の画像ファイル（JPG、PNGなど）

### 出力

- 検出結果を表示する画像ファイル（`{filename}.out.jpg`）
- `output_pth=True`の場合、生の検出結果（`{filename}.pth`）

## 例

入力画像`ycb.jpg`から物体を検出し、結果を`output/ycb.out.jpg`として保存します：

```bash
python ../demo.py --image_dir input --output_dir output
```

## 依存関係

- Python 3.7+
- PyTorch 1.9+
- torchvision
- Detectron2
- numpy
- matplotlib
- seaborn
- Pillow
- fire 