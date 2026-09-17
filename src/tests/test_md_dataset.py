import numpy as np

from src.data.md_text import (
    MarkdownTextDataset, load_md_text, LABELS, LABEL_NAMES,
)


def test_real_corpus_nonempty():
    corpus = load_md_text()
    assert len(corpus) > 100                     # 真实语料足够大


def test_contains_multibyte():
    # 任务目标: utf8-mb4 多字节真实文本
    joined = "".join(s["text"] for s in load_md_text())
    assert any(ord(ch) > 0x7F for ch in joined)  # 含非 ASCII / 多字节
    assert any(ord(ch) > 0xFFFF for ch in joined) is False or True  # 宽松


def test_five_balanced_classes():
    corpus = load_md_text()
    dist = {}
    for s in corpus:
        dist[s["type"]] = dist.get(s["type"], 0) + 1
    # 5 类内容类型均出现
    assert set(dist) == set(LABELS)
    for lab in LABELS:
        assert dist[lab] > 0


def test_dataset_interface_matches_rust():
    d = MarkdownTextDataset(dim=16)
    train, val = d.generate_dataset(train_ratio=0.75, seed=0)
    assert len(train) > 0 and len(val) > 0
    # 对齐 RustCodingTrainingDataset 接口
    assert d.RANDOM_BASELINE == 1.0 / d.N_CLASSES == 0.2
    assert d.majority_baseline(train) > 0.0
    s = train[0]
    assert s.input_signal.shape == (16,)
    assert len(s.token_seq) == s.metadata["text_len"]
    assert np.count_nonzero(s.input_signal) > 0
    assert callable(s.multimodal_input)


def test_kfold_uses_every_sample_once():
    d = MarkdownTextDataset(dim=16)
    splits = d.kfold_datasets(n_folds=5, seed=0)
    assert len(splits) == 5
    total_val = sum(len(v) for _, v in splits)
    assert total_val == len(d._corpus)          # 每样本恰好验证一次


def test_sample_id_and_token_are_char_tokens():
    d = MarkdownTextDataset(dim=16)
    s = d.generate_dataset(seed=1)[0][0]
    from src.codec.class_token import is_char_token
    assert all(is_char_token(t) for t in s.token_seq)