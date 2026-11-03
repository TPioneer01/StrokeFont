"""Launch local PyTorch workers using the current Python environment."""
import argparse
import os
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--master-port', type=int, default=0)
    parser.add_argument('script')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.nproc_per_node < 1:
        parser.error('--nproc-per-node must be positive')
    port = args.master_port
    if not port:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
    workers = []
    try:
        for rank in range(args.nproc_per_node):
            env = os.environ.copy()
            env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       RANK=str(rank), WORLD_SIZE=str(args.nproc_per_node),
                       LOCAL_RANK=str(rank))
            if args.nproc_per_node > 1:
                env.setdefault('OMP_NUM_THREADS', '1')
            workers.append(subprocess.Popen(
                [sys.executable, args.script, *args.script_args], env=env,
            ))
        while True:
            codes = [worker.poll() for worker in workers]
            failure = next((code for code in codes if code not in (None, 0)), None)
            if failure is not None:
                return failure
            if all(code is not None for code in codes):
                return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 130
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == '__main__':
    sys.exit(main())
