"""
Standalone smoke test for the full-line CRNN + CTC OCR pipeline
(backend/app/tasks/crnn_training.py).

This does NOT train a production model. It proves the pipeline is wired
correctly end-to-end -- model build, synthetic line rendering, CTC loss
computation/optimization, SavedModel export, TFLite conversion, TFLite
inference, and greedy CTC decode -- using a few dozen synthetic samples
and 3 epochs, all generated in-process (no DB, no labeled dataset, no
GPU required). It runs in a few seconds on CPU.

A real production model needs: a real annotated dataset (via the
project's annotation workflow / `train_crnn_model` Celery task in
crnn_training.py, which also pulls in real character crops + composited
lines from labeled plates), many more epochs, and ideally a GPU. Run
`train_crnn_model` (via the API's /ocr/train/crnn/{project_id} endpoint,
which enqueues it on Celery) for that -- this script is only a wiring
check, not a substitute for it.

Usage:
    cd backend && python3 scripts/smoke_test_crnn.py
"""
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.tasks.crnn_training as crnn  # noqa: E402


def main():
    rng = random.Random(1)
    n = 40
    texts = [crnn._random_string(rng, 4, 8) for _ in range(n)]
    imgs = []
    for t in texts:
        line = crnn._render_synthetic_line(t, rng)
        line = crnn._normalize_line(line)
        line = crnn._augment_line(line, rng)
        imgs.append(line)
    x = np.stack(imgs).astype(np.float32)[..., None] / 255.0

    labels = [crnn._encode(t) for t in texts]
    maxlen = max(len(l) for l in labels)
    y = np.full((n, maxlen), crnn.BLANK_INDEX, dtype=np.int32)
    label_lens = np.zeros((n, 1), dtype=np.int32)
    for i, l in enumerate(labels):
        y[i, : len(l)] = l
        label_lens[i, 0] = len(l)
    input_lens = np.full((n, 1), crnn.TIME_STEPS, dtype=np.int32)

    base = crnn._build_crnn(tf)
    labels_in = tf.keras.layers.Input(shape=(maxlen,), dtype="int32", name="labels")
    in_len_in = tf.keras.layers.Input(shape=(1,), dtype="int32", name="input_length")
    lab_len_in = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label_length")

    def ctc_lambda(args):
        lbl, y_pred, il, ll = args
        return tf.keras.backend.ctc_batch_cost(lbl, y_pred, il, ll)

    loss_out = tf.keras.layers.Lambda(ctc_lambda, name="ctc")(
        [labels_in, base.output, in_len_in, lab_len_in]
    )
    train_model = tf.keras.Model([base.input, labels_in, in_len_in, lab_len_in], loss_out)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), loss={"ctc": lambda yt, yp: yp}
    )

    t0 = time.time()
    hist = train_model.fit(
        [x, y, input_lens, label_lens], np.zeros(n), batch_size=8, epochs=3, verbose=2
    )
    print("Smoke-train elapsed sec:", round(time.time() - t0, 2))
    print("Loss history:", hist.history["loss"])
    assert hist.history["loss"][-1] < hist.history["loss"][0], "CTC loss did not decrease"

    out_dir = Path(tempfile.mkdtemp())
    export_dir = out_dir / "_saved_model_tmp"
    base.export(str(export_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path = out_dir / "ocr_crnn_smoketest.tflite"
    tflite_path.write_bytes(converter.convert())
    print("TFLite exported:", tflite_path, tflite_path.stat().st_size, "bytes")

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x[:1])
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
    decoded = crnn._greedy_decode(probs)
    print("Ground truth[0]:", texts[0], "| greedy-decoded (undertrained, expect noise):", decoded)
    print("SMOKE TEST PASSED (pipeline wiring only -- not a trained model).")


if __name__ == "__main__":
    main()
