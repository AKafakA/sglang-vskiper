"""Prove the process group we spawned owns the socket listening on a port.

`ss -ltnp` is not usable for this: on CSD3 it prints listening sockets with NO
process column at all, so a check that greps for pid= can never pass there --
the mirror image of a check that can never fail. It aborted a run whose warmup
had just succeeded 24/24.

This asks a different question. Instead of "who owns this socket globally",
which needs privileges, it asks "does any process in OUR group hold it", which
only needs to read our own /proc entries:

  1. read /proc/net/tcp{,6} for LISTEN rows on the port -> socket inodes
  2. walk the fds of every process in our process group
  3. a match proves we own the endpoint

Exit codes are distinct because the causes are: 0 owned, 3 a foreign listener
holds it, 4 nothing is listening. Only 0 may proceed.
"""
import os
import pathlib
import sys

OWNED, USAGE, FOREIGN, NOTHING = 0, 2, 3, 4


def listen_inodes(port: int) -> set[str]:
    want = f"{port:04X}"
    inodes = set()
    for name in ("tcp", "tcp6"):
        p = pathlib.Path("/proc/net") / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[1:]:
            f = line.split()
            if len(f) < 10:
                continue
            local, state, inode = f[1], f[3], f[9]
            if state != "0A":                      # 0A = TCP_LISTEN
                continue
            if local.rsplit(":", 1)[-1].upper() == want:
                inodes.add(inode)
    return inodes


def pids_in_group(pgid: int) -> list[int]:
    out = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) == pgid:
                out.append(int(entry.name))
        except (ProcessLookupError, PermissionError):
            continue
    return out


def group_holds(inodes: set[str], pgid: int) -> int | None:
    for pid in pids_in_group(pgid):
        fd_dir = pathlib.Path(f"/proc/{pid}/fd")
        try:
            fds = list(fd_dir.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return pid
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: prove_endpoint_owner.py <port> <pgid>", file=sys.stderr)
        return USAGE
    port, pgid = int(sys.argv[1]), int(sys.argv[2])
    inodes = listen_inodes(port)
    if not inodes:
        print(f"NOTHING-LISTENING port={port}")
        return NOTHING
    owner = group_holds(inodes, pgid)
    if owner is None:
        print(f"FOREIGN-LISTENER port={port} inodes={sorted(inodes)} pgid={pgid}")
        return FOREIGN
    print(f"OWNED port={port} pid={owner} pgid={pgid}")
    return OWNED


if __name__ == "__main__":
    sys.exit(main())
