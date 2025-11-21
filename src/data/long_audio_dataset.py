import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LongAudioDataset(Dataset):
    """
    Dataset that yields concatenated >10s audio samples plus metadata for chunking.
    """

    def __init__(self, hdf5_dir, transform=None, target_transform=None):
        self.hdf5_files = sorted(
            [f for f in os.listdir(hdf5_dir) if f.endswith(".h5")]
        )
        if not self.hdf5_files:
            raise ValueError(f"No HDF5 files found in {hdf5_dir}")
        self.hdf5_dir = hdf5_dir
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, index):
        file_name = self.hdf5_files[index]
        path = os.path.join(self.hdf5_dir, file_name)
        with h5py.File(path, "r") as h5:
            audio = h5["array"][:]
            text = h5["text"][()].decode("utf-8")
            duration = int(h5["duration"][()])
            seg_durs = h5["segment_durations"][:].astype(int)
            seg_text = [s.decode("utf-8") for s in h5["segment_text"][:]]

        if self._transform is not None:
            features, feat_len = self._transform(audio)
        else:
            raise ValueError("Transform must produce MFCC features and length")

        if self._target_transform is not None:
            target_tensor = self._target_transform(text)
        else:
            target_tensor = text

        meta = {
            "duration_ms": duration,
            "segment_durations_ms": seg_durs.tolist(),
            "segment_text": seg_text,
            "file_name": file_name,
            "raw_text": text,
        }

        return ((features, feat_len, meta), target_tensor)


def collate_long_input_sequences(samples):
    """
    Collate function that preserves metadata for each sample.
    """

    seqs = []
    seq_lens = []
    metas = []
    labels = []
    for (tensor, length, meta), label in samples:
        seqs.append(tensor)
        seq_lens.append(length)
        metas.append(meta)
        labels.append(label)

    batch_x = pad_sequence(seqs)
    batch_lens = torch.IntTensor(seq_lens)
    return (batch_x, batch_lens, metas), labels
