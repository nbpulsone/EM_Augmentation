import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")

    # augmentation-related arguments
    parser.add_argument("--student_lm", type=str, default="distilbert")
    parser.add_argument("--teacher_lm", type=str, default=None)
    parser.add_argument("--active_aug", action="store_true")
    parser.add_argument("--num_aug_examples", type=int, default=0)
    parser.add_argument("--selection_strategy", type=str, 
                        choices=["incorrect", "false_negative", 
                                 "uncertain", "low_confidence", "random"], 
                        default="incorrect")

    # active driver arugments
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--active_interval", type=int, default=3)
    parser.add_argument("--active_rounds", type=int, default=None, required=False)
    parser.add_argument("--c_eff", type=float, default=None, required=False)
    parser.add_argument("--max_aug_examples", type=int, default=None, required=False)
    parser.add_argument("--allow_repeat_anchors", dest="allow_repeat_anchors", action="store_true")

    # student model implementation details
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    if hp.fp16 and hp.bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")

    if (hp.fp16 or hp.bf16) and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires a CUDA GPU.")

    # validate augmentation configuration
    if hp.active_aug:
        if hp.warmup_epochs < 1:
            parser.error("--warmup_epochs must be >= 1")

        if hp.warmup_epochs >= hp.n_epochs:
            parser.error("Warmup leaves no augmentation opportunity")

        if hp.active_interval < 1:
            parser.error("--active_interval must be >= 1")

        if hp.active_rounds is not None and hp.active_rounds < 1:
            parser.error("--active_rounds must be >= 1")

        if hp.c_eff is not None and hp.c_eff <= 0:
            parser.error("--c_eff must be positive")

        if hp.max_aug_examples is not None and hp.max_aug_examples <= 0:
            parser.error("--max_aug_examples must be positive")

        if (
            hp.num_aug_examples <= 0
            and hp.c_eff is None
            and hp.max_aug_examples is None
        ):
            parser.error("Specify an augmentation budget")

        if hp.teacher_lm is None:
            hp.teacher_lm = "Qwen/Qwen2.5-7B-Instruct"

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_student-lm=%s_teacher-lm=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.student_lm, hp.teacher_lm,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag += (
        f"_active={hp.active_aug}"
        f"_strategy={hp.selection_strategy}"
        f"_numaug={hp.num_aug_examples}"
        f"_warmup={hp.warmup_epochs}"
        f"_interval={hp.active_interval}"
        f"_rounds={hp.active_rounds}"
        f"_ceff={hp.c_eff}"
        f"_maxaug={hp.max_aug_examples}"
        f"_repeat={hp.allow_repeat_anchors}"
    )
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.student_lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                   lm=hp.student_lm,
                                   max_len=hp.max_len,
                                   size=hp.size)
    valid_dataset = DittoDataset(validset, lm=hp.student_lm, max_len=hp.max_len)
    test_dataset = DittoDataset(testset, lm=hp.student_lm, max_len=hp.max_len)

    # train and evaluate the model   
    best_test_f1, best_val_f1, training_time, inference_time, num_test_pairs, inference_per_pair_ms = train(train_dataset, valid_dataset, test_dataset, run_tag, hp)
 
    print(f"~~~~ FINAL F1 RESULTS ({hp.task}) ~~~~")
    print(f"val f1 = {best_val_f1}")
    print(f"test f1 = {best_test_f1}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(f"~~~~ FINAL RUNTIME RESULTS ({hp.task}) ~~~~")
    print(f"Model = {hp.student_lm}")
    print(f"Training time (s) = {training_time:.4f}")
    print(f"Test inference time (s) = {inference_time:.4f}")
    print(f"Number of test pairs = {num_test_pairs}")
    print(f"Inference time per pair (ms) = {inference_per_pair_ms:.4f}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
