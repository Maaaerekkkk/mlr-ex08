# mlr-ex08 – Leaf Classifier

A web app that classifies leaves (apple, cherry, chestnut, maple) live from the phone's rear camera,
directly in the browser with [onnxruntime-web](https://onnxruntime.ai/).

**Page:** https://maaaerekkkk.github.io/mlr-ex08/

## Models
- `leaf_model_final.onnx` – Task 4 model (EfficientNet-B0, default on the page)
- `leaf_model.onnx` – Task 2 model (MobileNetV3-Small), selectable with the model switch

Both models share `model_info.json` (classes, input size, mean, std).

## Preprocessing (according to `model_info.json`)
1. centre square of the frame → resized to 96×96
2. RGB / 255 → 0–1
3. per-channel standardization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
4. tensor `1×3×96×96` (NCHW), output 4 logits → softmax

## Files
- `index.html` – the whole app (plain JavaScript)
- `leaf_model.onnx`, `leaf_model_final.onnx`, `model_info.json` – the models and their description
- `check_image.jpg`, `check_expected.json`, `check_expected_final.json` – check of both models (the "Test" button on the page)
- `vendor/jsquash-jpeg/` – mozjpeg JPEG decoder (Apache-2.0)
- `REVIEW.md` – review of the phone page (Task 2), `REPORT.md` – report of Task 4
- `task4_v2.py`, `make_splits.py`, `test_final_v2.py`, `results_v2.csv`, `test_results_v2.csv` and logs – Task 4 experiments

Works in Safari on iPhone (HTTPS, `playsinline`, single-threaded WASM). As a fallback, a photo can be uploaded.
JPEGs are decoded with mozjpeg (WASM) so that the pixels match PIL/libjpeg in every browser –
Safari's native decoder gives slightly different pixels and therefore different probabilities.
The image never leaves the phone.
