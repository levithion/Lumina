"""Optional public Hugging Face Dataset storage for indexed meme images."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from config import HF_DATASET_REPO, HF_TOKEN


def enabled() -> bool:
    return bool(HF_TOKEN and HF_DATASET_REPO)


def upload_image(local_path: str, object_name: str | None = None) -> str:
    if not enabled():
        return ""
    from huggingface_hub import HfApi

    repo_path = object_name or f"memes/{Path(local_path).name}"
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    owner, name = HF_DATASET_REPO.split("/", 1)
    return f"https://huggingface.co/datasets/{quote(owner)}/{quote(name)}/resolve/main/{quote(repo_path, safe='/')}"
