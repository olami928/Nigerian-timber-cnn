# Nigerian Timber ID website

The site is a static Vercel deployment. Identification runs in the visitor's
browser with the exported ShuffleNetV2 ONNX model; selected photographs are
not sent to a server. The model file is public and downloaded from this site.

## Build locally

From the repository root:

```sh
npm --prefix app ci
npm --prefix app run build
```

Serve `app/public/` with any static HTTP server. The build copies only the
ONNX export, class order, preprocessing configuration, and the single-threaded
ONNX Runtime Web WASM assets into the deployable folder. It regenerates the
species list and checks that the model classes, species order, and blur cutoff
match.

The model applies a minimum Laplacian-variance sharpness score of **171.4**,
the 5th percentile of training-image sharpness. Scores below this return a
retake message. No confidence retake threshold is set.

The Identify screen prominently reports independent-specimen image accuracy:
12.1% for ShuffleNetV2 and 11.3% for ResNet18 on 611 held-out images from 14
species. This is a research prototype with poor unseen-specimen performance,
not a validated timber identification service. Do not remove the limitation
notice without new evaluation and owner approval.

## Privacy and limitations

Images are decoded and processed locally in the browser and are not uploaded
or stored. The site does not log image contents. Browser compatibility,
mobile behavior, deployed-host behavior, and the 20-image notebook parity
check must be verified before deployment. Model predictions must not be used
for purchasing, legal, import/export, or structural-grading decisions.
