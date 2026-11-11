# mlr-ex08 – Klasifikátor listov

Webová aplikácia, ktorá živo klasifikuje listy (apple, cherry, chestnut, maple) zo zadnej kamery mobilu
pomocou modelu `leaf_model.onnx` a [onnxruntime-web](https://onnxruntime.ai/) priamo v prehliadači.

**Stránka:** https://maaaerekkkk.github.io/mlr-ex08/

## Predspracovanie (podľa `model_info.json`)
1. stredový štvorec snímky → zmenšenie na 96×96
2. RGB / 255 → 0–1
3. štandardizácia kanálov: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
4. tenzor `1×3×96×96` (NCHW), výstup 4 logity → softmax

## Súbory
- `index.html` – celá aplikácia (čistý JavaScript)
- `leaf_model.onnx`, `model_info.json` – model a jeho popis
- `check_image.jpg`, `check_expected.json` – kontrola (tlačidlo „Overiť check_image.jpg“ na stránke)

Funguje v Safari na iPhone (HTTPS, `playsinline`, jednovláknový WASM). Ako záloha je možné nahrať fotku.
JPEG sa dekóduje cez mozjpeg (WASM), aby sa pixely zhodovali s PIL/libjpeg vo všetkých prehliadačoch –
natívny dekodér Safari dáva mierne odlišné pixely a tým aj pravdepodobnosti.
