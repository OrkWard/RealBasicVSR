import warnings
import argparse
import glob
import os
import time
import threading
import psutil

warnings.filterwarnings("ignore", module="mmcv")
import cv2  # noqa: E402
import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmedit.core import tensor2img  # noqa: E402

from realbasicvsr.models.builder import build_model  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".mov")


class ResourceMonitor:
    """Background thread to monitor CPU, RAM, GPU usage and log to file."""

    def __init__(self, log_file, interval=0.5):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None
        self.process = psutil.Process()
        self.cuda_available = torch.cuda.is_available()

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        # Write header
        with open(self.log_file, "w") as f:
            if self.cuda_available:
                f.write(
                    "timestamp,cpu_percent,ram_mb,ram_percent,gpu_mem_mb,gpu_mem_percent,gpu_util_percent\n"
                )
            else:
                f.write("timestamp,cpu_percent,ram_mb,ram_percent\n")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                timestamp = time.time()

                # CPU and RAM usage
                cpu_percent = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                ram_mb = mem_info.rss / 1024 / 1024
                ram_percent = self.process.memory_percent()

                if self.cuda_available:
                    # GPU usage
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    gpu_mem_total = (
                        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    )
                    gpu_mem_percent = (gpu_mem_allocated / gpu_mem_total) * 100

                    # GPU utilization (from nvidia-smi)
                    try:
                        import subprocess

                        result = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=1,
                        )
                        gpu_util = float(result.stdout.strip())
                    except:
                        gpu_util = 0.0

                    # Write to log
                    with open(self.log_file, "a") as f:
                        f.write(
                            f"{timestamp:.3f},{cpu_percent:.1f},{ram_mb:.1f},{ram_percent:.2f},"
                            f"{gpu_mem_allocated:.1f},{gpu_mem_percent:.2f},{gpu_util:.1f}\n"
                        )
                else:
                    # Write CPU/RAM only
                    with open(self.log_file, "a") as f:
                        f.write(
                            f"{timestamp:.3f},{cpu_percent:.1f},{ram_mb:.1f},{ram_percent:.2f}\n"
                        )

                time.sleep(self.interval)
            except Exception as e:
                # Silently ignore errors in monitoring thread
                pass

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming inference script of RealBasicVSR (WSL2 optimized)"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("input_dir", help="directory of the input video")
    parser.add_argument("output_dir", help="directory of the output video")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=3,
        help="maximum sequence length to be processed (default 3 for WSL2)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of data loading workers (default 0 for WSL2)",
    )
    parser.add_argument(
        "--is_save_as_png", type=bool, default=True, help="whether to save as png"
    )
    parser.add_argument("--fps", type=float, default=25, help="FPS of the output video")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="enable benchmarking mode to show performance stats",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="performance.log",
        help="log file for resource usage monitoring",
    )
    args = parser.parse_args()

    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file."""
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config
    model.eval()

    return model


def load_video_frames(input_dir, num_workers=0):
    """Load video frames from file or directory.

    Args:
        input_dir: Path to video file or directory
        num_workers: If > 0, use multiprocessing for image loading
    """
    file_extension = os.path.splitext(input_dir)[1]
    inputs = []
    input_paths = []

    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(input_dir)
        for i, frame in enumerate(video_reader):
            inputs.append(np.flip(frame, axis=2).copy())
            input_paths.append(f"frame_{i:06d}")
    elif file_extension == "":
        input_paths = sorted(glob.glob(f"{input_dir}/*"))

        if num_workers > 0:
            # Parallel loading with multiprocessing
            from multiprocessing import Pool

            with Pool(num_workers) as pool:
                inputs = pool.map(
                    lambda p: mmcv.imread(p, channel_order="rgb"), input_paths
                )
        else:
            # Sequential loading
            for input_path in input_paths:
                img = mmcv.imread(input_path, channel_order="rgb")
                inputs.append(img)
    else:
        raise ValueError('"input_dir" can only be a video or a directory.')

    return inputs, input_paths


def process_and_save_streaming(model, frames, output_dir, input_paths, args):
    """Process frames in chunks and stream directly to output (no accumulation).

    This prevents memory accumulation by writing each chunk immediately.
    """
    device = next(model.parameters()).device
    num_frames = len(frames)
    max_seq_len = args.max_seq_len

    # Determine output mode
    is_video_output = os.path.splitext(output_dir)[1] in VIDEO_EXTENSIONS
    video_writer = None

    if args.benchmark:
        total_load_time = 0
        total_transfer_time = 0
        total_inference_time = 0
        total_save_time = 0
        start_time = time.time()

    print(f"Processing {num_frames} frames in chunks of {max_seq_len}...")

    for start_idx in range(0, num_frames, max_seq_len):
        end_idx = min(start_idx + max_seq_len, num_frames)
        chunk_size = end_idx - start_idx

        if args.benchmark:
            chunk_start = time.time()

        # Prepare chunk tensors
        chunk_inputs = []
        for i in range(start_idx, end_idx):
            img = torch.from_numpy(frames[i] / 255.0).permute(2, 0, 1).float()
            chunk_inputs.append(img.unsqueeze(0))
        chunk_tensor = torch.stack(
            [x.squeeze(0) for x in chunk_inputs], dim=0
        ).unsqueeze(0)

        if args.benchmark:
            load_time = time.time() - chunk_start
            total_load_time += load_time
            transfer_start = time.time()

        # Transfer to GPU
        chunk_tensor = chunk_tensor.to(device)

        if args.benchmark:
            transfer_time = time.time() - transfer_start
            total_transfer_time += transfer_time
            inference_start = time.time()

        # Run inference
        with torch.no_grad():
            output = model(chunk_tensor, test_mode=True)["output"]

        if args.benchmark:
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            save_start = time.time()

        # Move to CPU and save immediately (don't accumulate!)
        output_cpu = output.cpu()

        # Save this chunk
        if is_video_output:
            # Initialize video writer on first chunk
            if video_writer is None:
                h, w = output_cpu.shape[-2:]
                mmcv.mkdir_or_exist(os.path.dirname(output_dir))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_dir, fourcc, args.fps, (w, h))

            # Write frames from this chunk
            for i in range(output_cpu.size(1)):
                img = tensor2img(output_cpu[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
        else:
            # Save as images
            mmcv.mkdir_or_exist(output_dir)
            for i in range(output_cpu.size(1)):
                frame_idx = start_idx + i
                img = tensor2img(output_cpu[:, i, :, :, :])

                if frame_idx < len(input_paths):
                    filename = os.path.basename(input_paths[frame_idx])
                else:
                    filename = f"frame_{frame_idx:06d}.png"

                if args.is_save_as_png:
                    file_extension = os.path.splitext(filename)[1]
                    filename = filename.replace(file_extension, ".png")

                mmcv.imwrite(img, f"{output_dir}/{filename}")

        if args.benchmark:
            save_time = time.time() - save_start
            total_save_time += save_time

        # Free memory immediately
        del chunk_tensor, output, output_cpu
        torch.cuda.empty_cache()

        print(f"Processed frames {start_idx}-{end_idx-1} ({end_idx}/{num_frames})")

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        cv2.destroyAllWindows()

    if args.benchmark:
        total_time = time.time() - start_time
        print(f"\n=== Performance Statistics ===")
        print(f"Data loading time: {total_load_time:.3f}s")
        print(f"CPU->GPU transfer time: {total_transfer_time:.3f}s")
        print(f"Inference time: {total_inference_time:.3f}s")
        print(f"Save time: {total_save_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Frames processed: {num_frames}")
        print(f"FPS: {num_frames/total_time:.2f}")


def main():
    args = parse_args()

    # Start resource monitoring
    monitor = ResourceMonitor(args.log_file, interval=0.5)
    monitor.start()
    print(f"Resource monitoring started, logging to: {args.log_file}")

    start_time = time.time()

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available! Running on CPU will be very slow.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(
                f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # Initialize the model
        print("Loading model...")
        model_load_start = time.time()
        model = init_model(args.config, args.checkpoint)
        model = model.to(device)
        print(f"Model loaded in {time.time() - model_load_start:.2f}s")

        # Load video frames
        print(f"Loading video frames (num_workers={args.num_workers})...")
        load_start = time.time()
        frames, input_paths = load_video_frames(args.input_dir, args.num_workers)
        print(f"Loaded {len(frames)} frames in {time.time() - load_start:.2f}s")

        print(f"Using max_seq_len={args.max_seq_len}")

        # Process and save with streaming (no memory accumulation)
        process_and_save_streaming(model, frames, args.output_dir, input_paths, args)

        if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
            print(f"Video saved to: {args.output_dir}")
        else:
            print(f"Images saved to: {args.output_dir}")

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Overall FPS: {len(frames)/total_time:.2f}")

        if torch.cuda.is_available():
            print(
                f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )

    finally:
        # Stop monitoring
        monitor.stop()
        print(f"Resource monitoring stopped, log saved to: {args.log_file}")


if __name__ == "__main__":
    main()
