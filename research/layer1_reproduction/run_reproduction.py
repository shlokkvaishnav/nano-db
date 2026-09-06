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

# The bind mount, and the container-local working copy the experiment actually
# runs in. Amendment 1: the Windows/OneDrive bind mount is not a filesystem the
# substrate's assumptions hold on. MOUNT is for source in and results out; every
# build artifact, the corpus, and every run live in WORK.
MOUNT = "/repo"
WORK = "/work"

# CI's toolchain list (.github/workflows/ci.yml), so a build that works in CI
# works here -- PLUS the runtime the sweep needs and CI does not.
#
# Amendment 1(c): "exactly CI's list" was not enough, and the reason is
# structural rather than an oversight. CI builds and runs ctest; it never runs
# run_experiment.py. So nothing in CI ever imports `grpc`, and its list has no
# reason to carry the Python gRPC bindings. Without them every sweep cell dies
# in probe.ensure_stubs() and produces no samples.csv.
#
# python3-grpcio / python3-grpc-tools rather than pip: 24.04 marks the system
# environment externally managed, and pip would need --break-system-packages.
APT = ("cmake g++ libomp-dev protobuf-compiler protobuf-compiler-grpc "
       "libprotobuf-dev libgrpc++-dev libgrpc-dev pkg-config git python3 "
       "python3-pip python3-numpy curl ca-certificates "
       "python3-grpcio python3-grpc-tools rsync")

CMAKE = ("cmake -B build -DCMAKE_BUILD_TYPE=Release "
         "-DNANODB_BUILD_SERVER=ON -DNANODB_BUILD_CLUSTER=ON")

SEEDS = (20260800, 20260801, 20260802, 20260803, 20260804)

# run_experiment.py writes to a FIXED directory and has no flag to redirect it,
# which is why the top-level README's own recipe moves the output afterwards.
SRC_RESULTS = "research/replica_recall/results"
DEST_SWEEP = "research/layer1_reproduction/results_sweep"
CORPUS_CHECK = "research/layer1_reproduction/verify_corpus.py"


def sh(argv, dry, check=True):
    """Run one command as an ARGUMENT LIST, never through a shell.

    The first version built shell strings and quoted them with shlex.quote,
    under subprocess(shell=True). That is broken on the one host this harness
    exists for. `shell=True` on Windows runs cmd.exe, which does not treat
    single quotes as quoting at all, so:

      -v 'C:\\...\\repo':/repo   ->  docker splits on the colon in C: and dies
                                     with `invalid mode: /repo`
      bash -lc 'apt-get ...'     ->  bash receives the literal `'apt-get` alone

    The whole premise of this study is that the dev host is Windows and Docker
    is the only Linux available (README, "Why a container"), so a harness that
    only assembles commands correctly under a POSIX shell cannot run anywhere
    it is needed. Argument lists remove the quoting layer entirely rather than
    making it conditional on the platform.
    """
    print("  $ " + " ".join(shlex.quote(x) for x in argv), flush=True)
    if dry:
        return 0
    return subprocess.run(argv, check=check).returncode


def indocker(inner, dry, check=True, cwd=WORK):
    """Run one shell command inside the container.

    `inner` is a bash command line and stays a single string -- it is bash
    inside the container that parses it, so its quoting is POSIX by definition
    and correct. Only the OUTER docker invocation is a list.

    `cwd` defaults to WORK, the container's OWN filesystem, not the /repo bind
    mount. See Amendment 1 in SPEC.md: the mount is 7.9x slower for bulk writes,
    4.9x slower for metadata operations, and -- decisively -- does not give
    rename the atomicity the substrate assumes.
    """
    return sh(["docker", "exec", "-w", cwd, CONTAINER, "bash", "-lc", inner],
              dry, check)


def mount_source():
    """The host path in the form Docker's -v flag accepts.

    Docker Desktop takes `C:/Users/...`; backslashes are what get mangled once
    the value passes through any shell, so they are normalised away here rather
    than quoted around.
    """
    return ROOT.replace("\\", "/")


def ensure_container(dry):
    r = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{CONTAINER}$",
         "--format", "{{.Names}}"],
        capture_output=True, text=True)
    if CONTAINER in (r.stdout or ""):
        print(f"  container {CONTAINER} exists; starting if stopped")
        sh(["docker", "start", CONTAINER], dry, check=False)
        return
    # --network host is deliberate: the harness binds many local ports and the
    # corpus download needs egress. Docker Desktop may ignore it, which is
    # harmless here -- every port bound is reached from inside the container.
    sh(["docker", "run", "-d", "--name", CONTAINER, "--network", "host",
        "-v", f"{mount_source()}:/repo", "-w", "/repo", IMAGE,
        "sleep", "infinity"], dry)


def stage_deps(dry):
    print("\n=== deps ===")
    ensure_container(dry)
    indocker("apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
             f"apt-get install -y -qq {APT}", dry, cwd="/")
    # Fail here, loudly, rather than 10 sweep cells later. The sweep imports
    # grpc inside run_experiment.py, whose failure this harness deliberately
    # tolerates per-cell -- so a missing binding would otherwise surface as ten
    # empty runs and a zero exit code.
    indocker("python3 -c 'import grpc, grpc_tools.protoc' && "
             "echo 'grpc bindings present'", dry, cwd="/")
    # httplib is required by coordinator_main.cpp and server.cpp; pybind11 was
    # removed from .gitmodules, so this initialises exactly one submodule.
    # Done on the MOUNT, because that is where .gitmodules and the git dir live.
    indocker(f"git config --global --add safe.directory {MOUNT} && "
             "git submodule update --init --recursive", dry, cwd=MOUNT)
    sync_to_work(dry)


def sync_to_work(dry):
    """Copy the repo off the bind mount onto the container's own filesystem.

    Amendment 1. The experiment must not run on the mount: it is 7.9x slower for
    bulk writes and 4.9x slower for metadata operations, and `save_cluster_config`
    is not atomic there with respect to concurrent readers -- ClusterConfigRace
    fails 10/10 on the mount and passes 5/5 off it, with ~90,000 concurrent reads
    per run instead of 920.

    Excludes `build/`: an incremental tree configured with /repo paths would be
    reconfigured rather than reused, and CMake caches absolute paths.
    """
    print(f"  syncing {MOUNT} -> {WORK} (off the bind mount)")
    indocker(f"mkdir -p {WORK} && rsync -a --delete "
             f"--exclude build/ --exclude .fsbench/ "
             f"--exclude research/layer1_reproduction/results_sweep/ "
             f"{MOUNT}/ {WORK}/", dry, cwd="/")


def done_in_work(dest):
    """Has this sweep cell already produced output inside WORK?"""
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "test", "-f",
         f"{WORK}/{dest}/samples.csv"], capture_output=True)
    return r.returncode == 0


def sync_results_back(dry):
    """Copy the sweep's output back onto the mount, where git can see it.

    Only results travel back. Everything else in WORK is reproducible from the
    mount, and copying build artifacts back would put them on the slow
    filesystem for no reason.
    """
    print(f"  syncing results {WORK} -> {MOUNT}")
    indocker(f"mkdir -p {MOUNT}/{DEST_SWEEP} && "
             f"if [ -d {WORK}/{DEST_SWEEP} ]; then "
             f"rsync -a {WORK}/{DEST_SWEEP}/ {MOUNT}/{DEST_SWEEP}/; "
             f"else echo 'no results to sync'; fi", dry, cwd="/")


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
    produced = failed = skipped = 0
    for seed in SEEDS:
        for cond, flag, dur in (("baseline", "--no-chaos", 180),
                                ("chaos", "", 300)):
            dest = f"{DEST_SWEEP}/seed{seed}_{cond}"
            # Resumability check moved into the container: since Amendment 1 the
            # runs land in WORK, so testing the host path would never skip and
            # every resume would redo the whole sweep.
            if not dry and done_in_work(dest):
                print(f"  skip {dest} (exists)")
                skipped += 1
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
            if not dry:
                if done_in_work(dest):
                    produced += 1
                else:
                    failed += 1
                    print(f"  !! {dest} produced no samples.csv")
            # After each cell, not only at the end: a sweep interrupted at cell 7
            # should not lose the six runs before it.
            sync_results_back(dry)

    if not dry:
        print(f"\n  sweep cells: {produced} produced, {skipped} skipped "
              f"(already done), {failed} FAILED")
        if failed:
            # A sweep where every cell died used to exit 0, because each run is
            # tolerated individually so that one bad cell does not lose the
            # others. Tolerating each failure separately is right; reporting the
            # total as success is not -- it is the silent-failure shape #26,
            # #38, #46 and #48 were each caught by.
            raise SystemExit(
                f"{failed} of {produced + failed} sweep cells produced no data. "
                "The analysis would be over a partial sweep; fix the cause and "
                "re-run (completed cells are skipped).")


def stage_analyse(dry):
    print("\n=== analyse (committed tools, unmodified) ===")
    indocker(f"python3 research/replica_recall/aggregate.py "
             f"--sweep-dir {DEST_SWEEP} "
             f"| tee {DEST_SWEEP}/aggregate_output.txt", dry, check=False)
    sync_results_back(dry)


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
