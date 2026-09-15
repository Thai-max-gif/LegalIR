"""Select run deliverables, never the dataset, retrieval caches, or credentials."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import re
import sys
import tarfile


ROOT_FILES = (
    "submission.zip", "submission.json", "run_manifest.json", "checksums.sha256",
    "resolved_config.yaml", "dataset_manifest.json", "production_freeze.json",
    "kaggle_t4x2_report.json", "colab_t4_report.json", "environment.txt", "nvidia-smi.txt",
    "parameter_audit.json", "preflight_parameter_audit.json", "runtime_projection.json",
    "ablation_report.csv", "training.log",
)
MODEL_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt", "sentencepiece.bpe.model",
    "tokenizer.model", "training_manifest.json", "README.md",
)


def release_files(output_dir: Path) -> list[str]:
    root = Path(output_dir).resolve()
    paths = [root / name for name in ROOT_FILES]
    adapter = root / "checkpoints/reranker_final"
    paths.extend(adapter / name for name in MODEL_FILES)
    for pattern in ("checkpoints/fusion_final/*.json", "checkpoints/fusion_final/*.txt",
                    "cv/*report.json", "cv/fold_*/**/training_manifest.json",
                    "submissions/submission_manifest.json", "splits/*.json"):
        paths.extend(root.glob(pattern))
    selected = []
    for path in sorted(set(paths)):
        rel = path.relative_to(root)
        if any(part.startswith(".") or part in ("kaggle.json", "credentials.json") for part in rel.parts):
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise RuntimeError(f"Refusing symlink in release artifacts: {rel}")
        if path.is_file():
            selected.append(rel.as_posix())
    return selected


def build_recovery_archive(output_dir: Path) -> Path:
    """A bounded artifact archive can also be recovered after a failed upload."""
    target = output_dir / "recovery.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for name in release_files(output_dir):
            archive.add(output_dir / name, arcname=name, recursive=False)
    return target


@contextmanager
def training_log(output_dir: Path):
    """Tee complete lines, redacting credentials even when split across writes."""
    secrets = [value for key, value in os.environ.items()
               if value and ("TOKEN" in key or key in ("KAGGLE_KEY", "HF_TOKEN_WRITE", "HF_TOKEN_READ"))]

    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log, self.pending = stream, log, ""

        def emit(self, text):
            for secret in sorted(secrets, key=len, reverse=True):
                text = text.replace(secret, "[REDACTED]")
            text = re.sub(r"(?:hf_|KGAT_)[A-Za-z0-9_\-]+", "[REDACTED]", text)
            self.stream.write(text)
            self.log.write(text)
            self.stream.flush()
            self.log.flush()

        def write(self, text):
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self.emit(line + "\n")
            return len(text)

        def flush(self):
            self.stream.flush()
            self.log.flush()

        def finish(self):
            if self.pending:
                self.emit(self.pending)
                self.pending = ""

    with (output_dir / "training.log").open("w", encoding="utf-8") as log:
        stdout, stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
        except BaseException as exc:
            stderr.write(f"\nTraining interrupted: {type(exc).__name__}\n")
            raise
        finally:
            stdout.finish()
            stderr.finish()
