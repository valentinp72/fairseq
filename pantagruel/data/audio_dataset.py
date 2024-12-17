import logging
import torch
import time
import os
import numpy as np
import pandas as pd

from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from fairseq.data.data_utils import compute_block_mask_1d

logger = logging.getLogger(__name__)

class FileCSVAudioDataset(RawAudioDataset):
    def __init__(
        self,
        source_data_dir,
        samples_names_csv,
        sample_rate,
        audio_duration_seconds=30,
        audio_extension='flac',
        num_buckets=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask=False,
        **mask_compute_kwargs,
    ):
        self.num_frames = sample_rate * audio_duration_seconds
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=self.num_frames,
            min_sample_size=self.num_frames,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        self.fnames = pd.read_csv(samples_names_csv) \
            .apply(lambda row: os.path.join(
                source_data_dir,
                row.tarfile,
                f"{row.uuid}.{audio_extension}",
            ), axis=1).tolist()

        try:
            import pyarrow
            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.sizes = np.array([self.num_frames for _ in range(len(self))])
        self.set_bucket_info(num_buckets)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        import soundfile as sf
        path = self.fnames[index]
        path = path if isinstance(self.fnames, list) else path.as_py()

        retry = 3
        wav = None
        for i in range(retry):
            try:
                wav, curr_sample_rate = sf.read(
                    path,
                    dtype="float32",
                    frames=self.num_frames
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed to read {path}: {e}. Sleeping for {1 * i}"
                )
                time.sleep(1 * i)

        if wav is None:
            raise Exception(f"Failed to load {path}")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)

        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v


