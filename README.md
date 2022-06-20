# Quick start

#### Prepare data

Put your dataset into `./data` where contains `twi_emotion` already. You can follow the example (its formats) to construct your data. Download the other three benchmarking datasets (i.e., Topic, Situation and UnifyEmotion) from the existing entailment work [here](https://github.
com/yinwenpeng/BenchmarkingZeroShot). 

#### Start training

```python
pip install -r requirements.txt
python train_twi_emotion.py
# if you want to train on a new dataset
# create a training script following the examples provided, then
python train_<your_dataset>.py
```