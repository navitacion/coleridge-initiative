# Coleridge Initiative


Kaggle Competition Repogitory


## References

- [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)

- [Pytorch-CRF](https://pytorch-crf.readthedocs.io/en/stable/)

- [Jaccard-based FBeta score (micro F-0.5 score)](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)


## Team With

- [takapy](https://www.kaggle.com/takanobu0210)

- [asteriam](https://www.kaggle.com/masatakashiwagi)

**Thank you for playing with me!**


## Result

- Private Score: 0.084
- Rank: 766th / 1,610 (47.5%)

My Best Score

- Private Score: 0.218

Notebook is [here](https://www.kaggle.com/bootiu/coleridge-ner-crf-inference-ensemble-alltest?scriptVersionId=65925307)


## Getting Started

Easy to do, only type command.

```commandline
docker-compose up --build -d
```

Then, access below url
```
http://localhost:8888/lab
```

## Solution

- Preprocess
    - Add word tags for NER Training
        - We use 'BIOES' tags
            - 'O', 'B-NER', 'I-NER', 'E-NER', 'S-NER', 'PAD'
        - Created by takapy
            - https://www.kaggle.com/takanobu0210/coleridge-create-sequencelabeling-data
    - Delete row which contains > 2,000 words
    - Expand Test Data
        - BERT Input Max Length is up to 512 in the default setting.
        - So, we need to expand text data to input all of texts to BERT.
            - https://www.kaggle.com/bootiu/coleridge-expand-text-dataset
- Train Config
    - CV
        - StratifiedKFold(n_splits=5, shuffle=True)
        - StratifiedGroupKFold(n_splits=5, shuffle=True)
    - Batch Size
        - 4
    - Optimizer
        - RAdam (using pytorch_optimizer)
    - Scheduler
        - transformers.get_cosine_with_hard_restarts_schedule_with_warmup
- Model
    - SciBERT
        - backbone: SciBERT_scivocab_uncased
        - https://huggingface.co/allenai/scibert_scivocab_uncased
    - CRF
        - https://pytorch-crf.readthedocs.io/en/stable/
        

Model network is below.

```python
from torch import nn
import transformers
from torchcrf import CRF

class BERT_CRF_Model(nn.Module):
    def __init__(self, cfg, num_labels=6):
        super(BERT_CRF_Model, self).__init__()
        self.backbone = transformers.BertModel.from_pretrained(
            cfg.model,
            num_labels=num_labels
        )
        self.dropout = nn.Dropout(0.5)
        self.position_wise_ff = nn.Linear(768, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        out = self.dropout(out[0])
        out = self.position_wise_ff(out)
        
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(out, labels, reduction='mean'), self.crf.decode(out)
            log_likelihood = torch.abs(log_likelihood)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(out)
            return None, sequence_of_tags
```


## Notebook

- NER-CRF-benchmark.ipynb
    - Train & Logging & Inference
    - Using below
        - Pytorch Lightning
        - Comet_ML
        
**IMPORTANT:** Before executing, you must copy '.env.sample' ,rename to '.env' and describe your comet api key.

