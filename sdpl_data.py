import torch
import random
import os
import logging
import spdl
import spdl.io
import spdl.utils
import spdl.dataloader

from spdl.dataloader._task_runner import (
    apply_async,
    apply_concurrent,
)
from spdl.dataloader._utils import _iter_batch
import concurrent
# from spdl.dataset.imagenet import get_mappings, parse_wnid

def _init(debug, num_demux_threads, num_decode_threads, worker_id):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

def _get_batch_generator(args, device, path_class_pairs, indices):
    indices = indices[:((len(indices) // args.batch_size) * args.batch_size)]
    path_class_pairs = [path_class_pairs[i] for i in indices]
    # random.Random(42).shuffle(path_class_pairs)
    num_demux_threads = 12
    num_decode_threads = 12
    worker_id = 0
    debug = False
    _init(debug, num_demux_threads, num_decode_threads, worker_id)

    # train_dir = os.path.join(args.data_path, "train")
    srcs_gen = _iter_batch(
        path_class_pairs,
        batch_size=args.batch_size,
        # n=worker_id,
        # N=args.workers,
        # max=args.max_samples,
        # drop_last=True,
    )
    # import pdb; pdb.set_trace()

    # class_mapping, _ = get_mappings()

    async def _async_decode_func(paths):
        with torch.profiler.record_function("async_decode"):
            classes = [p[1] for p in paths]
            classes = torch.tensor(classes, dtype=torch.int64).to(device)
            image_paths = [p[0] for p in paths]
            buffer = await spdl.io.async_batch_load_image(
                image_paths,
                width=None,
                height=None,
                pix_fmt=None,
                strict=True,
                decode_options={
                    "filter_desc": "scale=width=256:height=256,crop=224:224,format=rgb24",
                },
                convert_options={
                    "cuda_device_index": getattr(args, 'gpu', 0),
                    "cuda_allocator": torch.cuda.caching_allocator_alloc,
                    "cuda_deleter": torch.cuda.caching_allocator_delete,
                },
            )
            batch = spdl.io.to_torch(buffer).to(torch.float32)
            batch = batch.permute((0, 3, 1, 2))
            return batch, classes

    @spdl.utils.chain_futures
    def _decode_func(paths):
        with torch.profiler.record_function("decode"):
            classes = [p[1] for p in paths]
            classes = torch.tensor(classes, dtype=torch.int64).to(device)
            image_paths = [p[0] for p in paths]
            buffer = yield spdl.io.batch_load_image(
                image_paths,
                width=None,
                height=None,
                pix_fmt=None,
                strict=True,
                decode_options={
                    "filter_desc": "scale=width=256:height=256,crop=224:224,format=rgb24",
                },
                convert_options={
                    "cuda_device_index": getattr(args, 'gpu', 0),
                    "cuda_allocator": torch.cuda.caching_allocator_alloc,
                    "cuda_deleter": torch.cuda.caching_allocator_delete,
                },
            )
            batch = spdl.io.to_torch(buffer).to(torch.float32)
            batch = batch.permute((0, 3, 1, 2))
            f = concurrent.futures.Future()
            f.set_result((batch, classes))
            yield f

    return apply_async(_async_decode_func, srcs_gen)
    # return apply_concurrent(_decode_func, srcs_gen)
    # match args.mode:
    #     case "concurrent":
    #         return apply_concurrent(_decode_func, srcs_gen)
    #     case "async":
    #         return apply_async(_async_decode_func, srcs_gen)
