"""
Data Loading Module for Weibo-COV Dataset

Handles loading and preprocessing of the Weibo-COV dataset for SV-NSDE training.

Dataset Schema (Weibo-COV):
    _id: Tweet identifier
    user_id: Hashed user identifier
    crawl_time: Collection timestamp
    created_at: Tweet creation time
    like_num: Like count
    repost_num: Retweet count
    comment_num: Comment count
    content: Tweet text
    origin_weibo: Original tweet reference (for retweets)
    geo_info: Geographic information
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CascadeData:
    """Represents a single information cascade."""
    cascade_id: str
    event_times: torch.Tensor      # Normalized timestamps [n_events]
    event_texts: List[str]         # Raw text content
    event_user_ids: List[str]      # User IDs
    metadata: Dict                  # Additional info (likes, reposts, etc.)

    @property
    def size(self) -> int:
        return len(self.event_times)

    @property
    def duration(self) -> float:
        if len(self.event_times) < 2:
            return 0.0
        return (self.event_times[-1] - self.event_times[0]).item()


class WeiboCOVLoader:
    """
    Loader for Weibo-COV dataset.

    Expected file format: CSV with columns:
        _id, user_id, crawl_time, created_at, like_num, repost_num,
        comment_num, content, origin_weibo, geo_info

    Usage:
        loader = WeiboCOVLoader("path/to/weibo_cov.csv")
        cascades = loader.build_cascades(min_size=10, max_size=1000)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        time_column: str = "created_at",
        text_column: str = "content",
        user_column: str = "user_id",
        cascade_column: str = "origin_weibo",
        time_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.data_path = Path(data_path)
        self.time_column = time_column
        self.text_column = text_column
        self.user_column = user_column
        self.cascade_column = cascade_column
        self.time_format = time_format

        self.df = None
        self.cascades = None

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load the CSV data."""
        logger.info(f"Loading data from {self.data_path}")

        self.df = pd.read_csv(
            self.data_path,
            nrows=nrows,
            parse_dates=[self.time_column],
            low_memory=False,
        )

        logger.info(f"Loaded {len(self.df):,} rows")
        return self.df

    def build_cascades(
        self,
        min_size: int = 5,
        max_size: int = 10000,
        time_unit: str = "hours",
    ) -> List[CascadeData]:
        """
        Build information cascades from retweet chains.

        Args:
            min_size: Minimum cascade size
            max_size: Maximum cascade size (truncate larger ones)
            time_unit: Time normalization unit ("hours", "minutes", "days")

        Returns:
            List of CascadeData objects
        """
        if self.df is None:
            self.load()

        # Group by original tweet (cascade root)
        logger.info("Building cascades...")

        # Filter retweets (have origin_weibo)
        retweets = self.df[self.df[self.cascade_column].notna()].copy()

        # Group by original tweet
        grouped = retweets.groupby(self.cascade_column)

        cascades = []
        for cascade_id, group in grouped:
            if len(group) < min_size:
                continue

            # Sort by time
            group = group.sort_values(self.time_column)

            # Truncate if too large
            if len(group) > max_size:
                group = group.head(max_size)

            # Extract timestamps
            times = pd.to_datetime(group[self.time_column])
            start_time = times.min()

            # Normalize to time units from start
            if time_unit == "hours":
                delta = (times - start_time).dt.total_seconds() / 3600
            elif time_unit == "minutes":
                delta = (times - start_time).dt.total_seconds() / 60
            else:  # days
                delta = (times - start_time).dt.total_seconds() / 86400

            event_times = torch.tensor(delta.values, dtype=torch.float32)

            # Extract texts
            event_texts = group[self.text_column].fillna("").tolist()

            # Extract user IDs
            event_user_ids = group[self.user_column].astype(str).tolist()

            # Metadata
            metadata = {
                "total_likes": group["like_num"].sum() if "like_num" in group else 0,
                "total_reposts": group["repost_num"].sum() if "repost_num" in group else 0,
                "total_comments": group["comment_num"].sum() if "comment_num" in group else 0,
                "unique_users": group[self.user_column].nunique(),
                "start_time": str(start_time),
            }

            cascade = CascadeData(
                cascade_id=str(cascade_id),
                event_times=event_times,
                event_texts=event_texts,
                event_user_ids=event_user_ids,
                metadata=metadata,
            )
            cascades.append(cascade)

        logger.info(f"Built {len(cascades)} cascades")
        self.cascades = cascades
        return cascades

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.cascades is None:
            raise ValueError("Call build_cascades() first")

        sizes = [c.size for c in self.cascades]
        durations = [c.duration for c in self.cascades]

        return {
            "num_cascades": len(self.cascades),
            "total_events": sum(sizes),
            "size_mean": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_median": np.median(sizes),
            "size_max": max(sizes),
            "duration_mean_hours": np.mean(durations),
            "duration_max_hours": max(durations),
        }

    def split_by_time(
        self,
        train_end: str = "2020-02-29",
        val_end: str = "2020-03-31",
    ) -> Tuple[List[CascadeData], List[CascadeData], List[CascadeData]]:
        """
        Split cascades by time period.

        As per the paper:
            - Outbreak period: Jan-Feb 2020 (high volatility)
            - Plateau period: Mar 2020
            - Decline period: Apr+ 2020 (low volatility)
        """
        if self.cascades is None:
            raise ValueError("Call build_cascades() first")

        train_cascades = []
        val_cascades = []
        test_cascades = []

        train_end_dt = pd.Timestamp(train_end)
        val_end_dt = pd.Timestamp(val_end)

        for cascade in self.cascades:
            start_time = pd.Timestamp(cascade.metadata["start_time"])

            if start_time <= train_end_dt:
                train_cascades.append(cascade)
            elif start_time <= val_end_dt:
                val_cascades.append(cascade)
            else:
                test_cascades.append(cascade)

        logger.info(f"Split: train={len(train_cascades)}, val={len(val_cascades)}, test={len(test_cascades)}")
        return train_cascades, val_cascades, test_cascades


def generate_synthetic_weibo_data(
    n_cascades: int = 1000,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data matching Weibo-COV schema for testing.

    This creates realistic-looking cascade data with:
    - Power-law distributed cascade sizes
    - Bursty temporal patterns
    - Synthetic Chinese-like content placeholders
    """
    np.random.seed(seed)

    records = []

    # Sample cascade sizes from power-law (Zipf)
    cascade_sizes = np.random.zipf(1.5, n_cascades)
    cascade_sizes = np.clip(cascade_sizes, 5, 500)

    base_time = datetime(2020, 1, 15, 0, 0, 0)

    for cascade_idx in range(n_cascades):
        cascade_id = f"cascade_{cascade_idx:06d}"
        n_events = int(cascade_sizes[cascade_idx])

        # Generate bursty inter-arrival times (mixture of exponentials)
        # This mimics the "staircase" growth pattern mentioned in the paper
        inter_arrivals = []
        current_rate = np.random.uniform(0.1, 2.0)

        for _ in range(n_events):
            # Occasionally have rate jumps (bursts)
            if np.random.random() < 0.1:
                current_rate = np.random.uniform(0.5, 5.0)

            dt = np.random.exponential(1.0 / current_rate)
            inter_arrivals.append(dt)

        times = np.cumsum(inter_arrivals)

        # Generate synthetic content (placeholder Chinese text)
        topics = [
            "疫情防控", "口罩", "武汉", "新冠病毒", "确诊病例",
            "隔离", "核酸检测", "疫苗", "健康码", "复工复产"
        ]
        sentiments = ["担心", "希望", "加油", "注意安全", "平安"]

        for event_idx in range(n_events):
            event_time = base_time + pd.Timedelta(hours=times[event_idx])

            # Synthetic content
            topic = np.random.choice(topics)
            sentiment = np.random.choice(sentiments)
            content = f"关于{topic}的讨论，{sentiment}！#COVID19#"

            # Engagement metrics (correlated with cascade position)
            position_factor = 1.0 - (event_idx / n_events)  # Earlier = more engagement

            records.append({
                "_id": f"{cascade_id}_{event_idx:04d}",
                "user_id": f"user_{np.random.randint(0, 100000):06d}",
                "crawl_time": str(datetime.now()),
                "created_at": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                "like_num": int(np.random.exponential(10) * position_factor),
                "repost_num": int(np.random.exponential(5) * position_factor),
                "comment_num": int(np.random.exponential(3) * position_factor),
                "content": content,
                "origin_weibo": cascade_id if event_idx > 0 else None,  # First is original
                "geo_info": np.random.choice(["北京", "上海", "武汉", "广州", None]),
            })

        # Shift base time for next cascade
        base_time += pd.Timedelta(hours=np.random.uniform(0.5, 24))

    df = pd.DataFrame(records)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")

    return df


class CascadeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for cascade data.

    Works with both real Weibo-COV data and synthetic data.
    """

    def __init__(
        self,
        cascades: List[CascadeData],
        tokenizer=None,
        max_events: int = 100,
        max_seq_length: int = 128,
        precomputed_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.cascades = cascades
        self.tokenizer = tokenizer
        self.max_events = max_events
        self.max_seq_length = max_seq_length
        self.precomputed_embeddings = precomputed_embeddings

    def __len__(self) -> int:
        return len(self.cascades)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cascade = self.cascades[idx]

        # Truncate if needed
        n_events = min(cascade.size, self.max_events)
        times = cascade.event_times[:n_events]
        texts = cascade.event_texts[:n_events]

        # Normalize times to [0, 1]
        if times.max() > 0:
            T = times.max().item()
            times = times / T
        else:
            T = 1.0

        result = {
            "times": times,
            "T": torch.tensor(T),
            "n_events": n_events,
            "cascade_id": cascade.cascade_id,
        }

        # Use precomputed embeddings if available
        if self.precomputed_embeddings and cascade.cascade_id in self.precomputed_embeddings:
            embeddings = self.precomputed_embeddings[cascade.cascade_id][:n_events]
            result["embeddings"] = embeddings
        elif self.tokenizer:
            # Tokenize on-the-fly
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            result["input_ids"] = encoded["input_ids"]
            result["attention_mask"] = encoded["attention_mask"]
        else:
            raise ValueError("Need either precomputed_embeddings or tokenizer")

        return result


def precompute_embeddings(
    cascades: List[CascadeData],
    encoder,
    tokenizer,
    batch_size: int = 32,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for all cascade events.

    This speeds up training by avoiding repeated BERT forward passes.
    """
    encoder = encoder.to(device)
    encoder.eval()

    embeddings_dict = {}

    for cascade in cascades:
        texts = cascade.event_texts
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                emb = encoder(encoded["input_ids"], encoded["attention_mask"])
                all_embeddings.append(emb.cpu())

        embeddings_dict[cascade.cascade_id] = torch.cat(all_embeddings, dim=0)

    if output_path:
        torch.save(embeddings_dict, output_path)
        logger.info(f"Saved embeddings to {output_path}")

    return embeddings_dict
