from zsl import DataArgs, WeaklyZSLTrainer, TrainingArgs
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    dataset_name = "twi_emotion"
    data_path = f"data/{dataset_name}"
    data_args = DataArgs(
        data_path=data_path,
        # max seq length of pseudo-labeled examples encoded for pre-training and unlabeled examples for self-training
        max_seq_length=128,
        # batch size in entailment and sent-embedding baseline runs
        bs=16,
        # newly added dataset apart from the benchmarking datasets: this is a multi-class classification task
        multi_label=False,
        # we use bert-base-uncased the base model in our system that can adapt to other transformers
        model_for_preselftrain="bert-base-uncased",
        model_for_fulltrain="bert-base-uncased",
        # sentence embedding model for label vocabulary generation
        model_for_labelvocab="deepset/sentence_bert",
    )
    training_args = TrainingArgs(
        output_path=data_path,
        selftrain_batch_size=128,
        pretrain_batch_size=16,
        # the accumulation_steps applies to pre-train, if there is no enough memory, increase it and reduce the bs
        accumulation_steps=1,
        pre_train_training_lr=5e-5,
        # options: linear, constant, linearconstant
        pretrain_lr_scheduler="linear",
        # True: override the pre-trained model if it exists already
        override=True,
        # this eval bs applies to pre-train and self-train
        eval_batch_size=128,
        # the following parameters refer to Yu et al., 2020
        pre_train_epochs=2,
        self_train_epochs=3,
        self_train_update_interval=50,
    )

    zsl_trainer = WeaklyZSLTrainer(data_args, training_args)
    # force=True: restart building the vocabulary even though it exists already
    # label expansion with label names and train (regarded as unlabeled) corpus: written to data/twi_emotion/label2vocab.json
    zsl_trainer.setup_labelvocab(force=False, corpus_from="train")
    # match and augment between train (regarded as unlabeled) corpus and label vocabulary - pseudo-labeled examples are written to: data/twi_emotion/train_bow_eda_td.json
    zsl_trainer.match_and_augment(target_set="train", expand_with_top=100, p=0.7, write_examples=True)
    # pseudo-labeled examples are written to: data/twi_emotion/train_bow_td.json
    # pre-train on pseudo-labeled examples and self-train on train (regarded as unlabeled)
    eval_results = zsl_trainer.pretrain_self_train(pretrain_set_name="train_bow_td", eval_set="test")
    print("-----------experimental results summary-----------")
    print(json.dumps(eval_results, indent=3))
