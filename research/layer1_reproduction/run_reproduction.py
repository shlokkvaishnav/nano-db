#!/usr/bin/env python3
"""
Reproduce the Layer-1 nano-db sweep in a Linux container, and commit its data (#53).

WHY A CONTAINER. The harness launches `build/nano_shard_node` and
`build/nano_coordinator` directly and its own docstring says "Run (Linux, after
building the binaries)". The development host is Windows/MinGW with no g++ and
no protoc, so the experiment cannot run there at all. Docker is the only Linux
available, which makes the container part of the measurement rather than an
implementation detail -- see SPEC.md's instrument characterization, and say so
in any writeup.

WHAT THIS DOES NOT DO: change the protocol. It runs the committed
`run_experiment.py` at the committed parameters and analyses with the committed
`analyze.py` / `aggregate.py`. A reproduction that alters the thing being
reproduced is a new experiment.

Stages are resumable and each skips work whose output already exists, because
the build is ~10 minutes and the corpus is ~186 MB; a failure in the sweep
should not force either to be redone.

    deps    apt packages + submodules
    build   cmake configure, build, ctest    <- ctest is the go/no-go gate
    corpus  fetch SIFT1M and verify it
    sweep   5 seeds x {baseline, chaos}
    analyse aggregate.py over the sweep

Usage (from the repo root, Docker running):
    python research/layer1_reproduction/run_reproduction.py --stage all
    python research/layer1_reproduction/run_reproduction.py --stage build
    python research/layer1_reproduction/run_reproduction.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

IMAGE = "ubuntu:24.04"
CONTAINER = "rrd-layer1"

# Exactly CI's toolchain list (.github/workflows/ci.yml), so a build that works
# in CI works here. Divergence between the two would be its own confound.
APT = ("cmake g++ libomp-dev protobuf-compiler protobuf-compiler-grpc "
       "libprotobuf-dev libgrpc++-dev libgrpc-dev pkg-config git python3 "
       "python3-pip python3-numpy curl ca-certificates")

CMAKE = ("cmake -B build -DCMAKE_BUILD_TYPE=Release "
         "-DNANODB_BUILD_SERVER=ON -DNANODB_BUILD_CLUSTER=ON")

SEEDS = (20260800, 20260801, 20260802, 20260803, 20260804)

# run_experiment.py writes to a FIXED directory and has no flag to redirect it,
# which is why the top-level README's own recipe moves the output afterwards.
SRC_RESULTS = "research/replica_recall/results"
DEST_SWEEP = "research/layer1_reproduction/results_sweep"
CORPUS_CHECK = "research/layer1_reproduction/verify_corpus.py"


def sh(cmd, dry, check=True):
    print(f"  $ {cmd}", flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, shell=True, check=check).returncode


def indocker(inner, dry, check=True):
    """Run one shell command inside the container, at the repo root."""
    return sh(f"docker exec -w /repo {CONTAINER} bash -lc {shlex.quote(inner)}",
              dry, check)


def ensure_container(dry):
    r = subprocess.run(
        f"docker ps -a --filter name=^{CONTAINER}$ --format {{{{.Names}}}}",
        shell=True, capture_output=True, text=True)
    if CONTAINER in (r.stdout or ""):
        print(f"  container {CONTAINER} exists; starting if stopped")
        sh(f"docker start {CONTAINER}", dry, check=False)
        return
    # --network host is deliberate: the harness binds many local ports and the
    # corpus download needs egress.
    sh(f"docker run -d --name {CONTAINER} --network host -v "
       f"{shlex.quote(ROOT)}:/repo -w /repo {IMAGE} sleep infinity", dry)


def stage_deps(dry):
    print("\n=== deps ===")
    ensure_container(dry)
    indocker("apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
             f"apt-get install -y -qq {APT}", dry)
    # httplib is required by coordinator_main.cpp and server.cpp; pybind11 was
    # removed from .gitmodules, so this initialises exactly one submodule.
    indocker("git config --global --add safe.directory /repo && "
             "git submodule update --init --recursive", dry)


def stage_build(dry):
    print("\n=== build (ctest is the go/no-go gate for outcome (iii)) ===")
    indocker(CMAKE, dry)
    indocker("cmake --build build -j$(nproc)", dry)
    indocker("cd build && ctest --output-on-failure", dry)


def stage_corpus(dry):
    print("\n=== corpus ===")
    indocker("python3 research/replica_recall/sift.py", dry, check=False)
    # Verified before use: a truncated download silently shrinks the corpus and
    # changes every number downstream.
    indocker(f"python3 {CORPUS_CHECK}", dry)


def stage_sweep(dry):
    """5 seeds x {baseline, chaos}, protocol unchanged.

    Durations match the README's published recipe: 180 s baseline, 300 s chaos.
    `--dist sift` with the default `--sift-vectors 200000` is the published
    corpus. Nothing here overrides a protocol parameter.
    """
    print("\n=== sweep: 5 seeds x baseline/chaos, protocol UNCHANGED ===")
    for seed in SEEDS:
        for cond, flag, dur in (("baseline", "--no-chaos", 180),
                                ("chaos", "", 300)):
            dest = f"{DEST_SWEEP}/seed{seed}_{cond}"
            if os.path.isdir(os.path.join(ROOT, dest)) and not dry:
                print(f"  skip {dest} (exists)")
                continue
            indocker(f"rm -rf {SRC_RESULTS}", dry, check=False)
            cmd = (f"python3 research/replica_recall/run_experiment.py "
                   f"--seed {seed} --dist sift --duration {dur}")
            if flag:
                cmd = f"{cmd} {flag}"
            indocker(cmd, dry, check=False)
            # Move rather than copy: leaving a populated results/ behind is how
            # #18 and #24 committed a stale run by accident.
            indocker(f"mkdir -p {DEST_SWEEP} && "
                     f"if [ -f {SRC_RESULTS}/samples.csv ]; then "
                     f"mv {SRC_RESULTS} {dest}; else "
                     f"echo NO_OUTPUT_seed{seed}_{cond}_run_failed; fi",
                     dry, check=False)


def stage_analyse(dry):
    print("\n=== analyse (committed tools, unmodified) ===")
    indocker(f"python3 research/replica_recall/aggregate.py "
             f"--sweep-dir {DEST_SWEEP} "
             f"| tee {DEST_SWEEP}/aggregate_output.txt", dry, check=False)


STAGES = {
    "deps": stage_deps, "build": stage_build, "corpus": stage_corpus,
    "sweep": stage_sweep, "analyse": stage_analyse,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | " + " | ".join(STAGES))
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command without running it")
    a = ap.parse_args()

    if a.stage != "all" and a.stage not in STAGES:
        print(f"unknown stage {a.stage!r}; choose from all, {', '.join(STAGES)}")
        return 2
    order = list(STAGES) if a.stage == "all" else [a.stage]

    print(f"repo: {ROOT}")
    print(f"image: {IMAGE}   container: {CONTAINER}")
    if a.dry_run:
        print("DRY RUN -- nothing will execute")
    for name in order:
        STAGES[name](a.dry_run)

    print("\nDry run complete." if a.dry_run else "\nDone.")
    print("Reminder: this is a reproduction on a NEW host, in a container the")
    print("original runs did not have. Agreement is evidence of reproducibility,")
    print("not a second confirmation of the effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
