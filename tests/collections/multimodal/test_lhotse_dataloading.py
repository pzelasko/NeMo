from itertools import islice
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.multimodal.data.audio_text_qa_dataset import TextProcessing
from nemo.collections.multimodal.data.audio_text_qa_dataset_lhotse import LhotseAudioQuestionAnswerDataset

lhotse = pytest.importorskip(
    "lhotse", reason="Lhotse + NeMo tests require Lhotse to be installed (pip install lhotse)."
)


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
    for c in cuts:
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def cutset_shar_path(cutset_path: Path) -> Path:
    """10 utterances of length 1s as a Lhotse Shar (tarred) CutSet."""
    from lhotse import CutSet

    cuts = CutSet.from_file(cutset_path)
    p = cutset_path.parent / "shar"
    p.mkdir(exist_ok=True)
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture()
def tokenizer() -> TokenizerSpec:
    class MockTokenizer(TokenizerSpec):
        """
        Inherit this class to implement a new tokenizer.
        """

        def text_to_tokens(self, text):
            return list(text)

        def tokens_to_text(self, tokens):
            return str(tokens)

        def tokens_to_ids(self, tokens):
            return list(map(ord, tokens))

        def ids_to_tokens(self, ids):
            return list(map(chr, ids))

        def text_to_ids(self, text):
            return self.tokens_to_ids(text)

        def ids_to_text(self, ids):
            return self.ids_to_tokens(ids)

    return MockTokenizer()


@pytest.fixture()
def text_processor(tokenizer: TokenizerSpec) -> TextProcessing:
    return TextProcessing(tokenizer=tokenizer)


def test_dataloader_from_lhotse_shar_cuts(cutset_shar_path: Path, text_processor: TextProcessing):
    config = OmegaConf.create(
        {
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "lhotse": {
                "shar_path": cutset_shar_path,
                "use_bucketing": True,
                "num_buckets": 2,
                "drop_last": False,
                "batch_duration": 4.0,  # seconds
                "quadratic_duration": 15.0,  # seconds
                "shuffle_buffer_size": 10,
                "buffer_size": 100,
                "seed": 0,
                "shar_seed": 0,
            },
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=LhotseAudioQuestionAnswerDataset(
            text_processor=text_processor,
            question="transcribe the following recording to text",
            tokens_to_generate=128,
            pad_to_max_length=False,
            max_seq_length=1024,
        ),
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3
