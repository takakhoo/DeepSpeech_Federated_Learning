# %%
import torch
import torch.optim as optim
import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath("../modules/deepspeech/src"))
from torchvision.transforms import Compose  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import torch.nn.functional as F
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

device = "cuda:0"

sys.path.insert(0, os.path.abspath("../src/"))

from models.ds1 import DeepSpeech1WithContextFrames  # noqa: E402
from ctc.ctc_loss_imp import batched_ctc_v2, ctc_loss_imp  # noqa: E402
from data.long_audio_dataset import LongAudioDataset, collate_long_input_sequences  # noqa: E402
from utils.plot import plot_four_graphs  # noqa: E402
from utils.util import init_a_point  # noqa: E402
from loss.loss import meta_loss, grad_distance, tv_norm  # noqa: E402


def get_device_net(FLAGS, use_relu):
    net = DeepSpeech1WithContextFrames(
        FLAGS.n_context, FLAGS.drop_prob, use_relu=use_relu
    ).to(device)
    return device, net


def zero_order_optimization_loop(
    inputs,
    x_param,
    output_sizes,
    target_size,
    net,
    dldw_targets,
    params_to_match,
    targets,
    prefix,
    FLAGS,
):
    net.eval()
    loss_func = lambda x, y: batched_ctc_v2(x, y, output_sizes, target_size)

    i = 0
    stop_condition = False
    tolerance = 10
    step_size = FLAGS.zero_lr

    while i < FLAGS.zero_max_iter and not stop_condition:
        directions = torch.randn(8, *x_param.shape).to(device)
        directions = directions.view(8, -1)
        directions = F.normalize(directions, dim=1)
        directions = directions.view(8, *x_param.shape)

        losses = []
        current_loss, _ = _chunk_meta_loss(
            net, x_param, targets, dldw_targets, params_to_match, loss_func, FLAGS
        )
        for d in directions:
            mloss, _ = _chunk_meta_loss(
                net,
                x_param + step_size * d,
                targets,
                dldw_targets,
                params_to_match,
                loss_func,
                FLAGS,
            )
            losses.append(mloss.item())

        best_direction = directions[int(torch.tensor(losses).argmin())]
        if min(losses) < current_loss.item():
            x_param = x_param + step_size * best_direction
            tolerance = 10
            step_size = FLAGS.zero_lr
        else:
            tolerance -= 1
            step_size *= 0.5
            logging.info(
                "[%s] Zero-order chunk=%s reducing step %.4f tol=%d",
                prefix,
                prefix,
                step_size,
                tolerance,
            )
            if tolerance < 0:
                stop_condition = True

        if i % 20 == 0:
            mae = torch.mean(torch.abs(x_param - inputs))
            logging.info(
                "[%s] zero-order iter=%d loss=%.5f mae=%.4f",
                prefix,
                i,
                min(losses),
                mae.item(),
            )
        i += 1
    return x_param


def _chunk_meta_loss(net, x_param, targets, dldw_targets, params_to_match, loss_func, FLAGS):
    out = net(x_param)
    out = out.log_softmax(-1)
    mloss, dldws = meta_loss(
        out, targets, None, None, dldw_targets, params_to_match, loss_func, FLAGS
    )
    return mloss, dldws


def first_order_optimization_loop(
    inputs,
    x_param,
    output_sizes,
    target_sizes,
    optimizer,
    scheduler,
    net,
    dldw_targets,
    params_to_match,
    targets,
    prefix,
    FLAGS,
):
    net.train()
    loss_func = lambda x, y: ctc_loss_imp(
        x, y, output_sizes, target_sizes, reduction="mean"
    )

    loss_history = []
    loss_gm_history = []
    loss_reg_history = []

    for i in range(FLAGS.max_iter):
        out = net(x_param).log_softmax(-1)
        mloss, dldws = meta_loss(
            out, targets, None, None, dldw_targets, params_to_match, loss_func, FLAGS
        )
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], FLAGS)
        gm_bias_distance = grad_distance(dldws[1], dldw_targets[1], FLAGS)

        if FLAGS.reg == "L2":
            regloss = torch.norm(x_param, p=2)
        elif FLAGS.reg == "TV":
            regloss = tv_norm(x_param.permute(1, 0, 2).unsqueeze(1))
        else:
            regloss = torch.tensor(0.0, device=device)

        loss = (1 - FLAGS.reg_weight) * mloss + FLAGS.reg_weight * regloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mae = torch.mean(torch.abs(x_param - inputs)).item()
        loss_history.append(loss.item())
        loss_gm_history.append(mloss.item())
        loss_reg_history.append(regloss.item())

        if i % 10 == 0:
            logging.info(
                "[%s] iter=%04d loss=%.6f gm=%.6f gw=%.6f gb=%.6f reg=%.6f grad=%.4f mae=%.4f",
                prefix,
                i,
                loss.item(),
                mloss.item(),
                gm_weight_distance.item(),
                gm_bias_distance.item(),
                regloss.item(),
                x_param.grad.norm().item(),
                mae,
            )
        if i % 200 == 0 and i > 0:
            plot_four_graphs(
                inputs.detach(),
                x_param.detach(),
                loss_history,
                loss_gm_history,
                loss_reg_history,
                i,
                prefix=prefix,
                FLAGS=FLAGS,
            )

    return x_param


def optimization_loop(
    inputs,
    x_param,
    output_sizes,
    target_sizes,
    optimizer,
    scheduler,
    net,
    dldw_targets,
    params_to_match,
    targets,
    prefix="",
    FLAGS=None,
):
    prefix_path = os.path.join(FLAGS.exp_path, f"{prefix}_x_param_first_order.pt")
    if not os.path.exists(prefix_path):
        logging.info("[%s] running first-order loop", prefix)
        x_param = first_order_optimization_loop(
            inputs,
            x_param,
            output_sizes,
            target_sizes,
            optimizer,
            scheduler,
            net,
            dldw_targets,
            params_to_match,
            targets,
            prefix,
            FLAGS,
        )
        torch.save(x_param.detach().cpu(), prefix_path)
    else:
        x_param = torch.load(prefix_path).to(device)
        logging.info("[%s] loaded previous first-order result", prefix)

    if FLAGS.zero_max_iter > 0:
        logging.info("[%s] starting zero-order refinement", prefix)
        x_param = zero_order_optimization_loop(
            inputs,
            x_param,
            output_sizes,
            target_sizes,
            net,
            dldw_targets,
            params_to_match,
            targets,
            prefix,
            FLAGS,
        )
    return x_param


def get_long_dataset(net, FLAGS):
    target_transform = Compose(
        [str.lower, net.ALPHABET.get_indices, torch.IntTensor]
    )
    dataset = LongAudioDataset(
        FLAGS.dataset_path, transform=net.transform, target_transform=target_transform
    )
    loader = DataLoader(
        dataset,
        collate_fn=collate_long_input_sequences,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        batch_size=FLAGS.batch_size,
        shuffle=False,
    )
    return dataset, loader


def encode_text(text, alphabet):
    text = text.lower()
    # Remove stray characters outside alphabet set
    filtered = "".join(ch for ch in text if ch in alphabet.symbols)
    return torch.LongTensor(alphabet.get_indices(filtered))


def compute_chunk_frames(meta, total_frames):
    total_ms = meta["duration_ms"]
    frames_per_ms = total_frames / max(total_ms, 1)
    frames = []
    start = 0
    for dur in meta["segment_durations_ms"]:
        end = start + max(1, int(round(dur * frames_per_ms)))
        frames.append((start, end))
        start = end
    if frames:
        frames[-1] = (frames[-1][0], total_frames)
    return frames


def reconstruct_long_sample(sample_idx, batch, network, FLAGS):
    (inputs, input_sizes, metas), _ = batch
    assert inputs.shape[1] == 1, "Batch size must be 1"
    total_frames = input_sizes[0].item()
    inputs = inputs[:total_frames].to(device)
    meta = metas[0]
    chunk_ranges = compute_chunk_frames(meta, total_frames)
    chunk_texts = meta["segment_text"]

    recon_chunks = []
    total_time = 0.0
    params_to_match = [
        network.network.out.module[0].weight,
        network.network.out.module[0].bias,
    ]
    chunk_metrics = []

    for chunk_idx, ((start, end), chunk_text) in enumerate(
        zip(chunk_ranges, chunk_texts)
    ):
        chunk_inputs = inputs[start:end].to(device)
        chunk_len = chunk_inputs.shape[0]
        if chunk_len == 0 or len(chunk_text.strip()) == 0:
            continue

        chunk_targets = encode_text(chunk_text, network.ALPHABET).unsqueeze(0).to(device)
        chunk_target_sizes = torch.LongTensor([chunk_targets.shape[1]]).to(device)
        chunk_input_sizes = torch.LongTensor([chunk_len]).to(device)

        output = network(chunk_inputs).log_softmax(-1)
        output_sizes = (
            torch.ones(output.shape[1], dtype=torch.int32, device=device) * output.shape[0]
        )
        loss_func = lambda x, y: batched_ctc_v2(x, y, output_sizes, chunk_target_sizes)
        loss = loss_func(output, chunk_targets)
        dldw_targets = torch.autograd.grad(loss, params_to_match)

        x_init = init_a_point(chunk_inputs, FLAGS)
        x_param = torch.nn.Parameter(x_init.to(device), requires_grad=True)
        if FLAGS.optimizer_name.lower() == "adam":
            optimizer = optim.Adam([x_param], lr=FLAGS.lr)
        else:
            optimizer = optim.SGD([x_param], lr=FLAGS.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(range(500, 4000, 500)), gamma=0.5
        )

        prefix = f"sampleidx_{sample_idx}_chunk_{chunk_idx}"
        chunk_targets_padded = chunk_targets

        start_time = time.time()
        x_param = optimization_loop(
            chunk_inputs,
            x_param,
            output_sizes,
            chunk_target_sizes,
            optimizer,
            scheduler,
            network,
            dldw_targets,
            params_to_match,
            chunk_targets_padded,
            prefix=prefix,
            FLAGS=FLAGS,
        )
        end_time = time.time()
        total_time += end_time - start_time

        save_path = os.path.join(
            FLAGS.exp_path, f"{prefix}_x_param_last.pt"
        )
        torch.save(
            {
                "x_param": x_param.detach().cpu(),
                "inputs": chunk_inputs.detach().cpu(),
                "targets": chunk_targets.cpu(),
                "transcript": chunk_text,
                "duration_ms": meta["segment_durations_ms"][chunk_idx],
                "time": end_time - start_time,
            },
            save_path,
        )

        with torch.no_grad():
            diff = (x_param.detach() - chunk_inputs).float()
            mse = diff.pow(2).mean().item()
            mae = diff.abs().mean().item()
            target_max = chunk_inputs.abs().max().item()
            psnr = (
                10 * torch.log10(torch.tensor((target_max**2) / (mse + 1e-12))).item()
                if mse > 0
                else float("inf")
            )

        chunk_metrics.append(
            {
                "chunk_idx": chunk_idx,
                "frame_start": start,
                "frame_end": end,
                "duration_ms": meta["segment_durations_ms"][chunk_idx],
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
                "text": chunk_text,
            }
        )

        recon_chunks.append(x_param.detach())

    if not recon_chunks:
        logging.warning("No valid chunks reconstructed for sample %d", sample_idx)
        return

    recon_full = torch.cat(recon_chunks, dim=0)
    gt_full = inputs[: recon_full.shape[0]].detach()
    full_path = os.path.join(FLAGS.exp_path, f"sampleidx_{sample_idx}_concat.pt")
    torch.save(
        {
            "x_param": recon_full.cpu(),
            "inputs": gt_full.cpu(),
            "transcript": meta["raw_text"],
            "time": total_time,
            "segment_text": chunk_texts,
            "segment_durations_ms": meta["segment_durations_ms"],
            "chunk_metrics": chunk_metrics,
            "duration_ms": meta["duration_ms"],
        },
        full_path,
    )

    plot_concat_summary(
        gt_full,
        recon_full,
        chunk_metrics,
        f"sampleidx_{sample_idx}_concat",
        FLAGS,
    )

    logging.info(
        "Finished sample %d (%.2fs audio) in %.1f seconds",
        sample_idx,
        meta["duration_ms"] / 1000,
        total_time,
    )


def plot_concat_summary(gt_tensor, recon_tensor, chunk_metrics, prefix, FLAGS):
    gt_np = gt_tensor.squeeze().cpu().numpy().T
    recon_np = recon_tensor.squeeze().cpu().numpy().T
    diff_np = (gt_tensor - recon_tensor).abs().squeeze().cpu().numpy().T

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, data, title in zip(
        (axes[0, 0], axes[0, 1], axes[1, 0]),
        (gt_np, recon_np, diff_np),
        ("Target MFCC", "Reconstruction MFCC", "Absolute Difference"),
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Feature")

    ax = axes[1, 1]
    if chunk_metrics:
        indices = [m["chunk_idx"] for m in chunk_metrics]
        mae_vals = [m["mae"] for m in chunk_metrics]
        ax.bar(indices, mae_vals, color="teal")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("MAE")
        ax.set_title("Per-chunk MAE")
    else:
        ax.text(0.5, 0.5, "No chunk metrics", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    fig_path = os.path.join(
        FLAGS.exp_path, "figures", f"{prefix}_summary_{stamp}.png"
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main(FLAGS):
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    fig_dir = os.path.join(FLAGS.exp_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    logging.info("Logging experiments to %s", FLAGS.exp_path)

    _, net = get_device_net(FLAGS, use_relu=False)

    dataset, loader = get_long_dataset(net, FLAGS)
    for sample_idx, batch in enumerate(loader):
        if sample_idx < FLAGS.batch_start:
            continue
        if FLAGS.batch_end != -1 and sample_idx >= FLAGS.batch_end:
            break
        reconstruct_long_sample(sample_idx, batch, net, FLAGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct >=10s audio by chunking and concatenation."
    )
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-end", type=int, default=4)
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--zero_lr", type=float, default=20.0)
    parser.add_argument(
        "--distance_function",
        type=str,
        default="cosine",
        choices=["L1", "L2", "cosine", "cosine+l2"],
    )
    parser.add_argument("--distance_function_weight", type=float, default=1.0)
    parser.add_argument(
        "--reg",
        type=str,
        default="None",
        choices=["L1", "L2", "TV", "None"],
    )
    parser.add_argument("--reg_weight", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=800)
    parser.add_argument("--zero_max_iter", type=int, default=80)
    parser.add_argument("--n_context", type=int, default=6)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--init_method", type=str, default="uniform")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/scratch2/f004h1v/datasets/librispeech_concat_10s",
    )
    parser.add_argument("--min_duration_ms", type=int, default=10000)
    parser.add_argument("--top_grad_percentage", type=float, default=1.0)

    FLAGS = parser.parse_args()
    assert FLAGS.batch_size == 1, "Batch size must be 1"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_bucket = f"{FLAGS.min_duration_ms//1000}s_plus"
    exp_base = os.path.join(project_root, "long_audio_runs", log_bucket)

    exp_name = (
        f"ds1_lr{FLAGS.lr}_iter{FLAGS.max_iter}_start{FLAGS.batch_start}_end{FLAGS.batch_end}"
    )
    FLAGS.exp_path = os.path.join(exp_base, exp_name)

    main(FLAGS)
