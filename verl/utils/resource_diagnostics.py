import datetime as _datetime
import faulthandler
import os
import signal
import sys
import threading
import traceback


_REGISTERED = False
_LOCK = threading.Lock()


def _now_tag():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _dump_torch_cuda(fh):
    try:
        import torch

        if not torch.cuda.is_available():
            fh.write("\n[torch.cuda] unavailable\n")
            return

        fh.write("\n[torch.cuda]\n")
        fh.write(f"is_initialized={torch.cuda.is_initialized()}\n")
        fh.write(f"device_count={torch.cuda.device_count()}\n")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            fh.write(
                f"device={idx} name={props.name} "
                f"allocated={torch.cuda.memory_allocated(idx) / 1024**3:.3f}GiB "
                f"reserved={torch.cuda.memory_reserved(idx) / 1024**3:.3f}GiB "
                f"max_allocated={torch.cuda.max_memory_allocated(idx) / 1024**3:.3f}GiB "
                f"max_reserved={torch.cuda.max_memory_reserved(idx) / 1024**3:.3f}GiB\n"
            )
        fh.write("\n[torch.cuda.memory_summary]\n")
        fh.write(torch.cuda.memory_summary(abbreviated=True))
        fh.write("\n")
    except Exception as exc:  # diagnostics must never crash training
        fh.write(f"\n[torch.cuda] dump failed: {exc!r}\n")


def dump_resource_diagnostics(signum=None, frame=None):
    log_dir = os.environ.get("RESOURCE_DIAGNOSTICS_DIR")
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"python_stack_pid{os.getpid()}_{_now_tag()}.log")

    with _LOCK:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"pid={os.getpid()} ppid={os.getppid()} signal={signum}\n")
            fh.write(f"argv={sys.argv!r}\n")
            fh.write(f"cwd={os.getcwd()}\n")
            fh.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
            fh.write(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}\n")
            fh.write(f"RESOURCE_DIAGNOSTICS_DIR={log_dir}\n")

            fh.write("\n[faulthandler]\n")
            faulthandler.dump_traceback(file=fh, all_threads=True)

            fh.write("\n[thread stacks]\n")
            frames = sys._current_frames()
            for thread in threading.enumerate():
                fh.write(f"\n--- thread name={thread.name} ident={thread.ident} daemon={thread.daemon} ---\n")
                thread_frame = frames.get(thread.ident)
                if thread_frame is not None:
                    traceback.print_stack(thread_frame, file=fh)

            _dump_torch_cuda(fh)


def register_resource_diagnostics():
    global _REGISTERED
    if _REGISTERED:
        return

    log_dir = os.environ.get("RESOURCE_DIAGNOSTICS_DIR")
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    faulthandler.enable(all_threads=True)

    try:
        signal.signal(signal.SIGUSR1, dump_resource_diagnostics)
    except Exception:
        return

    _REGISTERED = True
