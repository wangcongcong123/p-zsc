# Quick start

The repo contains the code for "Using {{Pseudo-Labelled Data}} for {{Zero-Shot Text Classification" and the paper can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-08473-7_4).


#### Prepare data

Put your dataset into `./data` where contains `twi_emotion` already. You can follow the example (its formats) to construct your data. Download the other three benchmarking datasets (i.e., Topic, Situation and UnifyEmotion) from the existing entailment work [here](https://github.com/yinwenpeng/BenchmarkingZeroShot). 

#### Start training

```python
pip install -r requirements.txt
python train_twi_emotion.py
# if you want to train on a new dataset
# create a training script following the examples provided, then
python train_<your_dataset>.py
```


### Citation

```
@inproceedings{Wang2022,
 title = {Using {{Pseudo-Labelled Data}} for {{Zero-Shot Text Classification}}},
 booktitle = {Proceedings of the 27th {{International Conference}} on {{Natural Language}} \& {{Information Systems}} ({{NLDB}} 2022)},
 author = {Wang, Congcong and Nulty, Paul and Lillis, David},
 year = {2022},
 month = {June},
 address = {{Valencia, Spain}},
 abstract = {Existing Zero-Shot Learning (ZSL) techniques for text classification typically assign a label to a piece of text by building a matching model to capture the semantic similarity between the text and the label descriptor. This is expensive at inference time as it requires the text paired with every label to be passed forward through the matching model. The existing approaches to alleviate this issue are based on exact-word matching between the label surface names and an unlabelled target-domain corpus to get pseudo-labelled data for model training, making them difficult to generalise to ZS classification in multiple domains, In this paper, we propose an approach called P-ZSC to leverage pseudo-labelled data for zero-shot text classification. Our approach generates the pseudo-labelled data through a matching algorithm between the unlabelled target-domain corpus and the label vocabularies that consist of in-domain relevant phrases via expansion from label names. By evaluating our approach on several benchmarking datasets from a variety of domains, the results show that our system substantially outperforms the baseline systems especially in datasets whose classes are imbalanced.},
}
```