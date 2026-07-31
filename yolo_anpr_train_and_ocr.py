"""
Repository: YOLO ANPR training + OCR
Files included below in a single document (each file begins with a clear header comment):

1) train_yolo_ultralytics.py    -> Recommended PyTorch (Ultralytics YOLOv8) training-from-scratch script
2) train_yolo_tensorflow_skeleton.py -> TensorFlow / Keras YOLO-like training skeleton (compact, train-from-scratch tiny-YOLO-like model)
3) inference_and_ocr.py         -> Load trained model (either Ultralytics .pt or TF saved_model), run detection, crop plates, run OCR (EasyOCR / Tesseract)

Usage: copy each section into its own .py file (filenames above). See the top of each section for requirements and quick run commands.

NOTE: This document intentionally keeps each script standalone and self-contained for ease of use. The Ultralytics script is the most production-ready and recommended if you're using PyTorch/GPU. The TensorFlow file is a compact skeleton you can expand on or replace with a full YOLO-TF implementation if preferred.

"""

###############################################################################
# FILE: train_yolo_ultralytics.py
# Recommended: Ultralytics YOLOv8 (PyTorch). Trains from scratch (random init) using a YAML model file.
###############################################################################

"""
Requirements:
    pip install ultralytics==8.*  # ultralytics YOLOv8 (PyTorch)
    pip install opencv-python pyyaml

Quick run (example):
    python train_yolo_ultralytics.py --data data.yaml --epochs 100 --batch 16 --img 640 --save_dir runs/exp1

Ensure your data.yaml points to absolute paths for train/val/test images and nc=1 names: ['license']
Example data.yaml:
train: E:/parking/newwork/train/images
val:   E:/parking/newwork/valid/images
test:  E:/parking/newwork/test/images
nc: 1
names: ['license']

This script will: create a YOLOv8n model from the YAML (random init), train for specified epochs, save best.pt and last.pt.
"""

# ------------------------------- CODE ---------------------------------------
import argparse
import os
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data.yaml', help='path to data.yaml')
    p.add_argument('--model', type=str, default='yolov8n.yaml', help='model yaml or name (we use yaml to initialize random weights)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--save_dir', type=str, default='runs/exp')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', type=str, default='0', help='cuda device or cpu')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize model from config (random init). Using 'yolov8n.yaml' that comes with ultralytics package.
    # Providing a local yaml path is fine. This will initialize randomly (no .pt pretrained).
    model = YOLO(args.model)  # random init if yaml specified and no weights file

    # Train. Ultralytics will save best.pt and last.pt in results folder (args.save_dir)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        workers=args.workers,
        device=args.device,
        project=os.path.dirname(args.save_dir) if os.path.dirname(args.save_dir) else '.',
        name=os.path.basename(args.save_dir),
        save=True,
        cache=False,
    )

    print(f"Training complete. Checkpoints and logs saved under: {args.save_dir}")


if __name__ == '__main__':
    main()

###############################################################################
# FILE: train_yolo_tensorflow_skeleton.py
# A compact TensorFlow / Keras YOLO-like training skeleton (tiny model).
# This is a lightweight starting point to train a YOLO-style detector from scratch in TF2.
# It is intentionally small so it trains quickly and demonstrates the end-to-end flow.
###############################################################################

"""
Requirements:
    pip install tensorflow==2.13.* opencv-python tensorflow-addons

Quick run (example):
    python train_yolo_tensorflow_skeleton.py --dataset_root "E:/parking/newwork" --epochs 100 --batch 16 --img_size 640

Notes:
 - This script does NOT implement the full-yolo head or advanced anchor logic.
 - It provides a trainable tiny YOLO-like RetinaNet/one-stage head to get you started training from random init.
 - For production-quality YOLO in TF, consider using an established TF implementation (e.g., YOLOv3/v4 TF ports) and replace the data pipeline.
"""

# ------------------------------- CODE ---------------------------------------
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


# Simple tiny backbone + detection head (single-scale, for demonstration only)
def build_tiny_yolo(input_shape=(640,640,3), num_classes=1):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    # final conv predicts (tx, ty, tw, th, obj_conf, class_probs)
    preds = keras.layers.Conv2D( (5 + num_classes), 1, padding='same')(x)
    # reshape to [batch, H*W, 5+num_classes]
    shape = tf.shape(preds)
    preds = tf.reshape(preds, (-1, shape[1]*shape[2], 5+num_classes))

    model = keras.Model(inputs, preds)
    return model


# Simple dataset generator for YOLO-format text files
class YoloDataset:
    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images = sorted(glob.glob(os.path.join(images_dir, '*')))
        self.labels_dir = labels_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.labels_dir, basename + '.txt')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # ground truth boxes from YOLO txt: class x_center y_center w h (normalized)
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, bw, bh = map(float, parts[:5])
                        boxes.append([cls, xc, yc, bw, bh])
        boxes = np.array(boxes, dtype=np.float32)
        # For this tiny demo, we'll produce a simple dense target of zeros.
        # Advanced target creation (anchors, grid assignment) is omitted here.
        target = np.zeros((1,))
        return img, target


def data_generator(images_dir, labels_dir, batch=8, img_size=640):
    ds = YoloDataset(images_dir, labels_dir, img_size=img_size)
    n = len(ds)
    i = 0
    while True:
        imgs = []
        targs = []
        for _ in range(batch):
            img, target = ds[i % n]
            imgs.append(img)
            targs.append(target)
            i += 1
        yield np.stack(imgs, axis=0), np.stack(targs, axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root', type=str, required=True, help='root folder containing train/valid/test with images/labels subfolders')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--save_dir', type=str, default='tf_runs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_images = os.path.join(args.dataset_root, 'train', 'images')
    train_labels = os.path.join(args.dataset_root, 'train', 'labels')
    val_images = os.path.join(args.dataset_root, 'valid', 'images')
    val_labels = os.path.join(args.dataset_root, 'valid', 'labels')

    model = build_tiny_yolo(input_shape=(args.img_size, args.img_size, 3), num_classes=1)
    model.summary()

    # Define simple loss: mean squared on raw preds (placeholder). Replace with proper detection losses for production.
    def simple_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred))

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=simple_loss)

    train_gen = data_generator(train_images, train_labels, batch=args.batch, img_size=args.img_size)
    val_gen = data_generator(val_images, val_labels, batch=args.batch, img_size=args.img_size)

    steps_per_epoch = max(1, len(os.listdir(train_images)) // args.batch)
    val_steps = max(1, len(os.listdir(val_images)) // args.batch)

    model.fit(train_gen,
              epochs=args.epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_gen,
              validation_steps=val_steps,
              callbacks=[
                  keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir, 'best_tf_yolo.h5'), save_best_only=True, monitor='val_loss'),
                  keras.callbacks.ReduceLROnPlateau(patience=5)
              ])

    model.save(os.path.join(args.save_dir, 'last_tf_yolo'))
    print('TF training complete. Models saved to', args.save_dir)


if __name__ == '__main__':
    main()

###############################################################################
# FILE: inference_and_ocr.py
# Detect plates with Ultralytics .pt OR a TF saved_model; crop plates and run OCR using EasyOCR or Tesseract.
###############################################################################

"""
Requirements:
    pip install ultralytics opencv-python easyocr pytesseract pillow

Quick run (Ultralytics detection example):
    python inference_and_ocr.py --source test_images/ --model_path runs/exp/best.pt --ocr easyocr --out_dir output

Notes:
 - If you trained with the TensorFlow skeleton, update the --tf_model path and the code branch will attempt to load a saved_model/h5.
 - OCR: EasyOCR typically gives better out-of-the-box results for license plates; Tesseract may require tuning (psm, oem, whitelist).
"""

# ------------------------------- CODE ---------------------------------------
import argparse
import os
import cv2
from PIL import Image
import numpy as np


def run_ultralytics_inference(model_path, image_path, conf_thres=0.25):
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(image_path, conf=conf_thres)
    # results is a Results object; supports .boxes
    crops = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crops.append((x1, y1, x2, y2))
    return crops


def run_tensorflow_inference(tf_model_path, image_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(tf_model_path)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img_rgb, (model.input.shape[1], model.input.shape[2]))
    inp = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)
    preds = model.predict(inp)
    # preds shape and decoding depend heavily on model architecture; here we return empty list as placeholder
    return []


def ocr_easyocr(crop_img):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    # convert to grayscale for easier OCR
    if isinstance(crop_img, np.ndarray):
        pil = Image.fromarray(crop_img)
    else:
        pil = crop_img
    img_np = np.array(pil)
    res = reader.readtext(img_np, detail=0)
    return res


def ocr_tesseract(crop_img):
    import pytesseract
    if isinstance(crop_img, np.ndarray):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2GRAY)
    # You may need to set tesseract_cmd if not in PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(gray, config=custom_config)
    return text.strip()


def crop_and_ocr(image_path, boxes, ocr_engine='easyocr'):
    img = cv2.imread(image_path)
    outputs = []
    for i, (x1,y1,x2,y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        # optional: resize plate to fixed height
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        scale = 120 / h
        crop_resized = cv2.resize(crop, (int(w*scale), 120))
        if ocr_engine == 'easyocr':
            texts = ocr_easyocr(crop_resized)
            text = texts[0] if texts else ''
        else:
            text = ocr_tesseract(crop_resized)
        outputs.append({'box': (x1,y1,x2,y2), 'text': text})
    return outputs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True, help='path to .pt (ultralytics) or tf saved_model')
    p.add_argument('--source', type=str, required=True, help='image or folder of images or video')
    p.add_argument('--ocr', type=str, default='easyocr', choices=['easyocr','tesseract'])
    p.add_argument('--out_dir', type=str, default='ocr_output')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', type=str, default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # source may be a file or folder
    if os.path.isdir(args.source):
        files = sorted([os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    else:
        files = [args.source]

    is_pt = args.model_path.endswith('.pt') or args.model_path.endswith('.pth')
    is_tf = os.path.isdir(args.model_path) or args.model_path.endswith('.h5')

    for f in files:
        if is_pt:
            boxes = run_ultralytics_inference(args.model_path, f, conf_thres=args.conf)
        elif is_tf:
            boxes = run_tensorflow_inference(args.model_path, f)
        else:
            print('Unknown model format for', args.model_path)
            boxes = []

        results = crop_and_ocr(f, boxes, ocr_engine=args.ocr)
        # save output image with boxes and text
        img = cv2.imread(f)
        for r in results:
            x1,y1,x2,y2 = r['box']
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, r['text'], (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        outpath = os.path.join(args.out_dir, os.path.basename(f))
        cv2.imwrite(outpath, img)
        # also print results
        print('File:', f)
        for r in results:
            print('  Box:', r['box'], 'Text:', r['text'])

    print('Done. Outputs in', args.out_dir)


if __name__ == '__main__':
    main()

###############################################################################
# END OF DOCUMENT
###############################################################################
