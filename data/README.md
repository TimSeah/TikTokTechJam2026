# Data

Datasets are downloaded locally and are not committed to Git. The primary expected layout is:

```text
data/
  cifake/
    train/{REAL,FAKE}/
    test/{REAL,FAKE}/
  validation_only/
    coco_val2017/
    dalle_advanced/
```

- Primary dataset: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Streaming fallback: [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set), labels 0 and 1 only
- Optional validation-only data: the exact COCO val2017 and DALL-E Advanced subset from WildFake

Training and evaluation scripts must consume seeded manifests. WildFake is never used for training.
