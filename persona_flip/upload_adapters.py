"""Upload the new seeds' LoRA adapters to one Hugging Face model repo, then check every file arrived intact (GPU box).

  python -m persona_flip.upload_adapters --repo <hf-user>/story-imprinting-qwen-seeds si27_s1_hb_dc si27_s1_hc_db
  python -m persona_flip.upload_adapters --self-test

Same layout as Kenney's adapter repo (one subfolder per fine-tune, adapter files directly inside), so his merge
script loads them: python -m src.merge_adapter --model qwen36_27b --run si27_s1_hb_dc --repo <repo>.
Per run, one commit with <qwen repo>/checkpoints/<run>/adapter/* and runs/seeds/train/<run>/{config.json,
train_<run>.jsonl} in <repo>/<run>/. Then each file's hash is compared with the Hub's (sha256 for LFS files, the
git blob id otherwise) and the result appended to runs/seeds/hf_upload.txt. Exits 1 if any file is missing or differs.
The repo is created private; make it public on the Hub once the base model's license is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from .common import CKPT, RUNS

SEEDS = RUNS / "seeds"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_id(data: bytes) -> str:
    """The id git (and the Hub) gives a non-LFS file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def run_files(run: str) -> dict[str, Path]:
    """path in the repo -> local file, for one fine-tune."""
    adapter = CKPT / run / "adapter"
    files = {f"{run}/{p.name}": p for p in sorted(adapter.glob("*")) if p.is_file()}
    for name in ("config.json", f"train_{run}.jsonl"):
        files[f"{run}/{name}"] = SEEDS / "train" / run / name
    missing = [str(p) for p in files.values() if not p.is_file()]
    if f"{run}/adapter_model.safetensors" not in files or missing:
        raise SystemExit(f"[upload] {run}: adapter or metadata missing ({adapter}; {missing})")
    return files


def check(api, repo: str, files: dict[str, Path]) -> list[str]:
    """Problems with the Hub's copy of these files (empty if every file matches the local one)."""
    remote = {f.path: f for f in api.get_paths_info(repo, list(files), expand=True)}
    problems = []
    for path, local in files.items():
        r = remote.get(path)
        if r is None:
            problems.append(f"{path}: not on the Hub")
        elif r.lfs is not None:
            if (r.lfs.size, r.lfs.sha256) != (local.stat().st_size, sha256(local)):
                problems.append(f"{path}: size or sha256 differs")
        elif r.blob_id != git_blob_id(local.read_bytes()):
            problems.append(f"{path}: blob id differs")
    return problems


def upload(repo: str, runs: list[str]) -> bool:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    ok = True
    for run in runs:
        files = run_files(run)
        commit = api.create_commit(repo, [CommitOperationAdd(path, str(p)) for path, p in files.items()],
                                   commit_message=f"Add {run} adapter")
        problems = check(api, repo, files)
        ok &= not problems
        lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} {repo}/{run} commit {commit.oid}: "
                 + (f"{len(files)} files match" if not problems else "MISMATCH " + "; ".join(problems))]
        lines += [f"  {path} {p.stat().st_size} {sha256(p)}" for path, p in files.items()]
        with (SEEDS / "hf_upload.txt").open("a") as f:
            f.write("\n".join(lines) + "\n")
        print("[upload] " + lines[0])
    return ok


def self_test() -> None:
    import tempfile

    assert git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"   # git hash-object of an empty file
    assert git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"

    class Lfs:
        def __init__(self, size, sha):
            self.size, self.sha256 = size, sha

    class Remote:
        def __init__(self, path, blob_id=None, lfs=None):
            self.path, self.blob_id, self.lfs = path, blob_id, lfs

    with tempfile.TemporaryDirectory() as d:
        big, small = Path(d) / "a.safetensors", Path(d) / "c.json"
        big.write_bytes(b"\0" * 5000)
        small.write_bytes(b"{}\n")
        files = {"r/a.safetensors": big, "r/c.json": small}

        class Api:
            def __init__(self, rows):
                self.rows = rows

            def get_paths_info(self, repo, paths, expand):
                return [r for r in self.rows if r.path in paths]

        good = [Remote("r/a.safetensors", lfs=Lfs(5000, sha256(big))), Remote("r/c.json", git_blob_id(b"{}\n"))]
        assert check(Api(good), "x", files) == []
        assert len(check(Api(good[:1]), "x", files)) == 1                                  # file missing on the Hub
        assert len(check(Api([Remote("r/a.safetensors", lfs=Lfs(5000, "0" * 64)), good[1]]), "x", files)) == 1
        assert len(check(Api([good[0], Remote("r/c.json", git_blob_id(b"{ }\n"))]), "x", files)) == 1
    print("[upload] self-test: 6 checks pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="e.g. si27_s1_hb_dc")
    ap.add_argument("--repo")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.repo and args.runs):
        raise SystemExit("--repo and at least one run are required")
    raise SystemExit(0 if upload(args.repo, args.runs) else 1)


if __name__ == "__main__":
    main()
