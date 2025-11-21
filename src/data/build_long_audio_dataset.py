import argparse
import os
import glob
import h5py
import numpy as np
from datetime import datetime


def load_sample(h5_path):
    with h5py.File(h5_path, "r") as h5:
        audio = h5["array"][:]
        text = h5["text"][()].decode("utf-8")
        duration = int(h5["duration"][()])
        sampling_rate = int(h5["sampling_rate"][()])
    return {
        "audio": audio,
        "text": text,
        "duration": duration,
        "sampling_rate": sampling_rate,
        "path": os.path.basename(h5_path),
    }


def save_concat_sample(output_dir, idx, segments):
    audio = np.concatenate([seg["audio"] for seg in segments])
    text = " ".join(seg["text"].strip() for seg in segments).replace("  ", " ")
    duration = sum(seg["duration"] for seg in segments)
    sampling_rate = segments[0]["sampling_rate"]

    seg_durations = np.array([seg["duration"] for seg in segments], dtype=np.int32)
    seg_lengths = np.array([len(seg["text"]) for seg in segments], dtype=np.int32)
    seg_paths = np.array(
        [seg["path"] for seg in segments],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    seg_texts = np.array(
        [seg["text"] for seg in segments],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    out_path = os.path.join(output_dir, f"dataset_item_{idx}.h5")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("array", data=audio, compression="gzip", compression_opts=3)
        h5.create_dataset("text", data=text.encode("utf-8"))
        h5.create_dataset("duration", data=duration)
        h5.create_dataset("sampling_rate", data=sampling_rate)
        h5.create_dataset("segment_durations", data=seg_durations)
        h5.create_dataset("segment_text_lengths", data=seg_lengths)
        h5.create_dataset("segment_file_paths", data=seg_paths)
        h5.create_dataset("segment_text", data=seg_texts)

    return out_path, duration


def build_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = sorted(
        glob.glob(os.path.join(args.source_dir, "*.h5")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    if not all_files:
        raise ValueError(f"No HDF5 files found in {args.source_dir}")

    segments = []
    running_dur = 0
    num_written = 0

    for fp in all_files:
        sample = load_sample(fp)
        segments.append(sample)
        running_dur += sample["duration"]

        if running_dur >= args.target_duration_ms:
            save_concat_sample(args.output_dir, num_written, segments)
            num_written += 1
            segments = []
            running_dur = 0

            if args.max_samples and num_written >= args.max_samples:
                break

    if running_dur >= args.min_trailing_ms and segments:
        save_concat_sample(args.output_dir, num_written, segments)
        num_written += 1

    print(
        f"[{datetime.now()}] Wrote {num_written} samples >= {args.target_duration_ms/1000:.1f}s to {args.output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate short LibriSpeech samples into >=10s HDF5 entries."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing the original short-form HDF5 samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for concatenated samples.",
    )
    parser.add_argument(
        "--target-duration-ms",
        type=int,
        default=10000,
        help="Minimum duration (in ms) for each concatenated sample.",
    )
    parser.add_argument(
        "--min-trailing-ms",
        type=int,
        default=6000,
        help="Allow writing a final example if leftover duration exceeds this threshold.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of concatenated samples to write (0 = no cap).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
