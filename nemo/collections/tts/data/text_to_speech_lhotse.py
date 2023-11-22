import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Literal

import torch.utils.data

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.preprocessing.features import Featurizer
from nemo.utils import logging


class LhotseTextToSpeechDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    def __init__(
        self,
        text_tokenizer: BaseTokenizer,
        speaker_path: Optional[Path] = None,
        featurizers: Optional[Dict[str, Featurizer]] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        align_prior_hop_length: Optional[int] = None,
        audio_norm: Literal["none", "volume", "loudness"] = "volume",
        loudness_norm_dbfs: float = -12,
    ):
        from lhotse.dataset import AudioSamples

        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.align_prior_hop_length = align_prior_hop_length
        self.audio_norm = audio_norm
        assert self.audio_norm in (
            "none",
            "volume",
            "loudness",
        ), f"Unsupported value: {audio_norm=} (we support 'none', 'volume', and 'loudness')"
        self.loudness_norm_dbfs = loudness_norm_dbfs

        self.speaker_index_map = None
        if speaker_path:
            with open(speaker_path, 'r', encoding="utf-8") as speaker_f:
                self.speaker_index_map = json.load(speaker_f)

        self.featurizers = []
        if featurizers:
            logging.info(f"Found featurizers {featurizers.keys()}")
            self.featurizers.extend(featurizers.values())

        self.feature_processors = []
        if feature_processors:
            logging.info(f"Found featurize processors {feature_processors.keys()}")
            self.feature_processors.extend(feature_processors.values())

        self.load_audio = AudioSamples(fault_tolerant=True)

    @property
    def include_speaker(self) -> bool:
        return self.speaker_index_map is not None

    @property
    def include_align_prior(self) -> bool:
        return self.align_prior_hop_length is not None

    def __getitem__(self, cuts) -> Dict[str, Union[torch.Tensor, List[str]]]:
        from lhotse.dataset.collation import collate_vectors

        cuts = cuts.sort_by_duration()

        if self.audio_norm == "loudness":
            cuts = cuts.normalize_loudness(self.loudness_norm_dbfs)
        audio, audio_lens, cuts = self.load_audio(cuts)
        if self.audio_norm == "volume":
            audio = 0.95 * audio / audio.abs().max(dim=1, keepdims=True).values

        def fetch_text(cut) -> str:
            supervision = cut.supervisions[0]
            if supervision.custom is not None and "normalized_text" in supervision.custom:
                return supervision.normalized_text
            return supervision.text

        tokens = [torch.as_tensor(self.text_tokenizer(fetch_text(c))) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.text_tokenizer.pad)

        ans = {
            # TODO(pzelasko): we can get dataset name from the cut if we attach it earlier
            #                 either during dataset creation or dynamically, if we know each data source's name
            #                 we can do that in nemo.collections.common.data.lhotse.cutset
            # "dataset_names": ...,
            # TODO(pzelasko): if we sample from Lhotse Shar dataset there is not really a filepath
            #                 but if we read from a "normal" manifest we can pass "cut.recording.sources[0].source"
            # "audio_filepaths": ...,
            "audio": audio,
            "audio_lens": audio_lens,
            "text": tokens,
            "text_lens": token_lens,
        }

        if self.include_speaker:
            ans.update(
                speaker=[cut.supervisions[0].speaker for cut in cuts],
                speaker_index=[self.speaker_index_map[cut.supervisions[0].speaker] for cut in cuts],
            )

        if self.include_align_prior:
            raise NotImplementedError()
            # feature_dict = featurizer.load(
            #     manifest_entry=data.manifest_entry, audio_dir=data.audio_dir, feature_dir=data.feature_dir
            # )
            # example.update(feature_dict)

        for processor in self.feature_processors:
            raise NotImplementedError()
            # processor.process(example)

        return ans


def _identity(x):
    return x
