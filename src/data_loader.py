import logging
import os
import pickle
import tarfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = BASE_DIR / "data_cache"


@st.cache_data(show_spinner=False)
def _get_cached_registry(registry_path: str) -> pd.DataFrame:
    return pd.read_csv(registry_path)


def prepare_test_data_locally() -> pd.DataFrame:
    img_dir = DATA_CACHE_DIR / "test_images"
    registry_path = DATA_CACHE_DIR / "test_registry.csv"

    if registry_path.exists():
        return _get_cached_registry(str(registry_path))

    placeholder = st.empty()
    with placeholder.container():
        st.info(
            "📦 First launch: Preparing images... Check your terminal for progress."
        )

        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        archive_path = DATA_CACHE_DIR / "cifar.tar.gz"

        # 1. Loading
        if not archive_path.exists():
            logger.info("Downloading archive from University of Toronto...")
            response = requests.get(url, stream=True, timeout=30)
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete.")

        # 2. Unpacking
        logger.info("Extracting test_batch...")
        with tarfile.open(archive_path, "r:gz") as tar:
            member = tar.getmember("cifar-10-batches-py/test_batch")
            member.name = os.path.basename(member.name)
            tar.extract(member, path=DATA_CACHE_DIR)
        archive_path.unlink()
        logger.info("Extraction complete.")

        # 3. Conversion
        logger.info("Converting to JPG...")
        with open(DATA_CACHE_DIR / "test_batch", "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            images = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            filenames = entry["filenames"]
            labels = entry["labels"]

        data_records = []
        p_bar = st.progress(0)
        for i in range(len(images)):
            img_name = f"{filenames[i]}.jpg"
            full_path = img_dir / img_name

            if not full_path.exists():
                Image.fromarray(images[i]).save(full_path)

            rel_path = f"data_cache/test_images/{img_name}"
            data_records.append({"image_path": rel_path, "label": int(labels[i])})

            if i % 1000 == 0:
                p_bar.progress((i + 1) / 10000)
                logger.info(f"Processed {i}/10000 images...")

        df = pd.DataFrame(data_records)
        df.to_csv(registry_path, index=False)
        logger.info("Registry saved successfully!")
        st.success("✅ Done!")
        time.sleep(1)

    placeholder.empty()
    return df


def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        full_path = (
            BASE_DIR / image_path if not os.path.isabs(image_path) else Path(image_path)
        )
        if full_path.exists():
            return Image.open(full_path).convert("RGB")
        return None
    except Exception:
        return None
