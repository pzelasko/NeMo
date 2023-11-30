from typing import Optional

import torch.utils.data

from nemo.collections.multimodal.data.audio_text_qa_dataset import TextProcessing
from nemo.collections.multimodal.parts.utils.data_utils import ceil_to_nearest


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        question: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        noise_cuts: Optional = None,
    ):
        from lhotse.dataset import AudioSamples, CutMix

        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.maybe_mix_noise = (
            _identity if noise_cuts is None else CutMix(noise_cuts, pad_to_longest=False, random_mix_offset=True)
        )
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        # TODO(pzelasko): This is a placeholder for the actual question injection design.
        #                 For now, make it work with a single question and consult the
        #                 actual design with Tom and Steve.
        self.question = question

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()
        cuts = self.maybe_mix_noise(cuts)

        audio, audio_lens, cuts = self.load_audio(cuts)

        collated_text_data = collate_text_data(
            cuts=cuts,
            question=self.question,
            text_processor=self.text_processor,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
        )

        return {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,
            "audio_signal_length": audio_lens,
            "metadata": {"cuts": cuts.map(drop_data).to_eager()},
            "audio_ratio": torch.ones(audio.shape[0], dtype=torch.float32),
            **collated_text_data,
        }


def collate_text_data(
    cuts,
    question: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    from lhotse.dataset.collation import collate_vectors
    from .audio_text_qa_dataset import _build_loss_mask

    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        adjust_input_ids(text_processor._process_example(context=question, output=cut.supervisions[0].text))
        for cut in cuts
    ]
    fields = as_dict(examples)

    all_tokens = collate_vectors(fields["input_ids"], padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    max_length = max(
        len(x) + len(y) + len(z) + tokens_to_generate
        for x, y, z in zip(fields["input_ids"], fields["context_ids"], fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))
    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors([_build_loss_mask(item)[1:] for item in examples], padding_value=0),
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], padding_value=pad_id),
        "context_lengths": torch.LongTensor(fields["context_lengths"]),
        "answers": collate_vectors(fields["answers"], padding_value=pad_id),
        "max_length": torch.LongTensor(max_length),
    }


def adjust_input_ids(item: dict) -> dict:
    """Mimics the logic from nemo/collections/multimodal/data/audio_text_qa_dataset.py:131"""
    item["input_ids"] = item.get("masked_input_ids", item["input_ids"])
    return item


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}


def drop_data(cut):
    """Removes in-memory data from a Lhotse cut; useful for returning metadata-only cuts from dataloader."""
    from lhotse.shar.writers.shar import to_shar_placeholder

    cut.recording = to_shar_placeholder(cut.recording, cut)
    return cut


def _identity(x):
    return x
