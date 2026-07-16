"""
voting.py — Phase-3 self-consistency voting: cluster executed values with
tolerance, weighted vote, consensus gating for the (optional) judge layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    value: float
    program: Optional[str]
    strategy_id: str
    sample_index: int          # 0 = greedy sample of its strategy
    stated_result: Optional[float] = None

    @property
    def is_greedy(self) -> bool:
        return self.sample_index == 0


@dataclass
class VoteConfig:
    abs_tol: float = 1e-4
    rel_tol: float = 0.005          # 0.5% for larger magnitudes
    greedy_weight_bonus: float = 0.5
    stated_agree_bonus: float = 0.25
    accept_confidence: float = 0.5  # top cluster weight share
    accept_min_members: int = 3
    accept_margin: float = 0.15     # (top - second) / total


@dataclass
class Cluster:
    rep_value: float
    weight: float = 0.0
    members: list[Candidate] = field(default_factory=list)


@dataclass
class VoteOutcome:
    value: Optional[float]
    confidence: float
    margin: float
    clusters: list[Cluster]
    needs_judge: bool
    reason: str  # "consensus" | "low_confidence" | "low_margin" | "few_members" | "no_candidates"


def _tolerance(a: float, b: float, cfg: VoteConfig) -> float:
    return max(cfg.abs_tol, cfg.rel_tol * max(abs(a), abs(b)))


def _candidate_weight(c: Candidate, cfg: VoteConfig) -> float:
    weight = 1.0
    if c.is_greedy:
        weight += cfg.greedy_weight_bonus
    if c.stated_result is not None and abs(c.stated_result - c.value) <= _tolerance(
        c.stated_result, c.value, cfg
    ):
        weight += cfg.stated_agree_bonus
    return weight


def cluster_and_vote(candidates: list[Candidate], cfg: VoteConfig | None = None) -> VoteOutcome:
    cfg = cfg or VoteConfig()

    valid = [
        c for c in candidates
        if c.value is not None and c.value == c.value and abs(c.value) < 1e15
    ]
    if not valid:
        return VoteOutcome(None, 0.0, 0.0, [], True, "no_candidates")

    # Single-link sweep over sorted values.
    valid.sort(key=lambda c: c.value)
    clusters: list[Cluster] = []
    for cand in valid:
        if clusters and abs(cand.value - clusters[-1].rep_value) <= _tolerance(
            cand.value, clusters[-1].rep_value, cfg
        ):
            cluster = clusters[-1]
            cluster.members.append(cand)
            cluster.weight += _candidate_weight(cand, cfg)
            # Representative = weighted median approximation: keep the member
            # value closest to the running mean of the cluster.
            mean = sum(m.value for m in cluster.members) / len(cluster.members)
            cluster.rep_value = min(cluster.members, key=lambda m: abs(m.value - mean)).value
        else:
            clusters.append(Cluster(rep_value=cand.value, weight=_candidate_weight(cand, cfg),
                                    members=[cand]))

    clusters.sort(key=lambda c: c.weight, reverse=True)
    total_weight = sum(c.weight for c in clusters)
    top = clusters[0]
    second_weight = clusters[1].weight if len(clusters) > 1 else 0.0
    confidence = top.weight / total_weight if total_weight else 0.0
    margin = (top.weight - second_weight) / total_weight if total_weight else 0.0

    # Final value: prefer a greedy member's value inside the winning cluster.
    greedy_members = [m for m in top.members if m.is_greedy]
    final_value = greedy_members[0].value if greedy_members else top.rep_value

    if confidence < cfg.accept_confidence:
        return VoteOutcome(final_value, confidence, margin, clusters, True, "low_confidence")
    if len(top.members) < cfg.accept_min_members:
        return VoteOutcome(final_value, confidence, margin, clusters, True, "few_members")
    if margin < cfg.accept_margin:
        return VoteOutcome(final_value, confidence, margin, clusters, True, "low_margin")
    return VoteOutcome(final_value, confidence, margin, clusters, False, "consensus")
