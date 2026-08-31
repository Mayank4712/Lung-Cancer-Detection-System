# notebooks/

Training notebooks (e.g. a Colab/Kaggle GPU notebook for the classifier and
segmenter) are maintained outside this repository because they are heavyweight
and tied to Colab environments.

They are not required to run the demo: the backend starts successfully in demo
mode without trained `.pth` weights (see `TRAINING.md`).

Local CPU training can be smoke-tested via:

```bash
cd backend
python scripts/train_classifier.py --dry-run --epochs 1 --limit 32
python scripts/train_segmenter.py --dry-run --epochs 1 --limit 32
```
