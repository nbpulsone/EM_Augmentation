import hashlib
import json
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .error_selector import select_examples
from .teacher import TeacherModel

logger = logging.getLogger(__name__)


class ActiveAugmentationDriver:
    """Closed-loop Active Data Augmentation (ADA) orchestrator.

    Wraps selection (`select_examples`), LLM counterfactual synthesis (`TeacherModel`),
    and dynamic dataset expansion (`train_dataset.add_new_examples`) with:
      - Warm-up and periodic round scheduling (`should_run`)
      - Without-replacement anchor tracking across rounds
      - Fractional augmentation budget enforcement (C_eff = N_syn / N_a)
      - Empty/malformed teacher output guards
      - Per-round telemetry for F1-vs-budget evaluation curves

    Note on data leakage:
        `feedback_dataset` should be the training positive-anchor pool or a
        dedicated feedback split carved out of training—never the evaluation
        validation split used for early stopping or checkpoint selection.
    """

    def __init__(self, hp: Any) -> None:
        self.hp = hp
        self.teacher = TeacherModel(hp.teacher_lm or TeacherModel.DEFAULT_TEACHER)

        # Optional ADA hyperparameters (safe defaults if not yet defined on `hp`)
        self.default_num_aug: int = int(getattr(hp, "num_aug_examples", 0))
        self.warmup_epochs: int = int(getattr(hp, "warmup_epochs", 3))
        self.active_interval: int = int(getattr(hp, "active_interval", 2))
        self.max_rounds: Optional[int] = getattr(hp, "active_rounds", None)
        self.target_c_eff: Optional[float] = getattr(hp, "c_eff", None)
        self.max_aug_examples: Optional[int] = getattr(hp, "max_aug_examples", None)
        self.allow_repeat_anchors: bool = bool(getattr(hp, "allow_repeat_anchors", False))

        if self.target_c_eff is not None and self.target_c_eff > 1.0:
            logger.warning(
                "Configured c_eff=%.2f exceeds 1.0; adding more synthetic negatives "
                "than positive anchors risks negative-class swamping.",
                self.target_c_eff,
            )

        # Cross-round state
        self.round_idx: int = 0
        self.total_augmented: int = 0
        self.initial_positive_anchors: Optional[int] = None
        self.augmented_anchor_keys: Set[str] = set()
        self.history: List[Dict[str, Any]] = []

    def should_run(self, epoch: int) -> bool:
        """Return True if an active round should trigger at the given 1-indexed epoch."""
        if self.max_rounds is not None and self.round_idx >= self.max_rounds:
            return False
        if epoch < self.warmup_epochs:
            return False
        if self.active_interval > 1 and (epoch - self.warmup_epochs) % self.active_interval != 0:
            return False
        if self.max_aug_examples is not None and self.total_augmented >= self.max_aug_examples:
            return False
        if (
            self.target_c_eff is not None
            and self.initial_positive_anchors
            and self.total_augmented >= math.floor(self.target_c_eff * self.initial_positive_anchors)
        ):
            return False
        return True

    def run_round(
        self,
        train_dataset: Any,
        feedback_dataset: Any,
        evaluation_details: Any,
        epoch: Optional[int] = None,
    ) -> List[Any]:
        """Execute a single closed-loop active data augmentation round.

        Args:
            train_dataset: Active training dataset implementing `add_new_examples(examples)`.
            feedback_dataset: Candidate pool used to diagnose student uncertainty/errors.
            evaluation_details: Student predictions/metrics from the latest sweep.
            epoch: Optional 1-indexed epoch number; if provided, enforces `should_run(epoch)`.

        Returns:
            List of valid newly synthesized training examples added in this round.
        """
        if epoch is not None and not self.should_run(epoch):
            logger.debug("Skipping active round at epoch %s (outside schedule or budget).", epoch)
            return []

        if self.max_rounds is not None and self.round_idx >= self.max_rounds:
            logger.info("Skipping active round: reached max_rounds=%d.", self.max_rounds)
            return []

        if feedback_dataset is None:
            logger.warning("Skipping active round: `feedback_dataset` is None.")
            return []

        # Lazily snapshot seed positive anchor count N_a before any mutation
        if self.initial_positive_anchors is None:
            self.initial_positive_anchors = self._count_positive_anchors(train_dataset)

        round_budget, remaining_budget = self._resolve_round_budget()
        if round_budget <= 0:
            logger.info(
                "Skipping active round: augmentation budget exhausted (total_augmented=%d).",
                self.total_augmented,
            )
            return []

        # 1. Select informative candidates from student weaknesses / uncertainty
        selected = select_examples(feedback_dataset, evaluation_details, self.hp)
        if not selected:
            logger.info("Round %d: selector returned no candidates.", self.round_idx + 1)
            self._record_round(epoch, 0, 0, round_budget, 0)
            return []

        # 2. Enforce without-replacement sampling across rounds
        candidates, candidate_keys = self._filter_candidates(selected, limit=round_budget)
        if not candidates:
            logger.info(
                "Round %d: all %d selected candidates were already augmented.",
                self.round_idx + 1,
                len(selected),
            )
            self._record_round(epoch, len(selected), 0, round_budget, 0)
            return []

        # 3. Teacher synthesizes targeted counterfactual examples
        num_to_request = min(round_budget, len(candidates)) if not self.allow_repeat_anchors else round_budget
        if remaining_budget is not None:
            num_to_request = min(num_to_request, remaining_budget)

        teacher_outputs = self.teacher.generate_examples(candidates, num_to_request)
        if not teacher_outputs:
            logger.warning("Round %d: teacher returned no outputs.", self.round_idx + 1)
            self._record_round(epoch, len(selected), len(candidates), num_to_request, 0)
            return []

        # TEMP debug: see teacher output
        for output in teacher_outputs[:3]:
            print("Raw teacher output:", repr(output))

        # 4. Convert teacher outputs to valid EM training examples
        augmented_examples = self.teacher.build_augmented_examples(teacher_outputs)
        augmented_examples = [ex for ex in (augmented_examples or []) if ex is not None]
        print(
            f"Teacher generation: "
            f"requested={num_to_request}, "
            f"returned={len(teacher_outputs)}, "
            f"parsed={len(augmented_examples)}"
        )

        # require distinct pairs across rounds/original training set examples
        existing_pairs = {(left.lower(), right.lower()) for left, right in train_dataset.pairs}
        unique_examples = []

        for left, right, label in augmented_examples:
            key = (left.lower(), right.lower())
            if key in existing_pairs:
                continue
            existing_pairs.add(key)
            unique_examples.append((left, right, label))

        augmented_examples = unique_examples[:num_to_request]
        print(
            f"Teacher generation after deduplication: "
            f"accepted={len(augmented_examples)}"
        )

        # Enforce hard total budget cap even if teacher over-generated
        if remaining_budget is not None and len(augmented_examples) > remaining_budget:
            augmented_examples = augmented_examples[:remaining_budget]

        if not augmented_examples:
            logger.warning(
                "Round %d: no valid augmented examples remained after building/validation.",
                self.round_idx + 1,
            )
            self._record_round(epoch, len(selected), len(candidates), num_to_request, 0)
            return []

        # 5. Update active training dataset and driver state
        train_dataset.add_new_examples(augmented_examples)

        if not self.allow_repeat_anchors:
            self.augmented_anchor_keys.update(candidate_keys)

        added_count = len(augmented_examples)
        self.total_augmented += added_count
        self.round_idx += 1

        self._record_round(epoch, len(selected), len(candidates), num_to_request, added_count)
        logger.info(
            "Active round %d complete (epoch=%s): added=%d, total_augmented=%d, c_eff=%s",
            self.round_idx,
            epoch,
            added_count,
            self.total_augmented,
            f"{self.get_effective_c():.3f}" if self.get_effective_c() is not None else "N/A",
        )
        return augmented_examples

    def get_effective_c(self) -> Optional[float]:
        """Compute current effective fractional augmentation ratio C_eff = N_syn / N_a."""
        if not self.initial_positive_anchors or self.initial_positive_anchors <= 0:
            return None
        return self.total_augmented / float(self.initial_positive_anchors)

    def get_summary(self) -> Dict[str, Any]:
        """Return run-level telemetry for logging and F1-vs-budget plots."""
        return {
            "rounds_completed": self.round_idx,
            "total_augmented": self.total_augmented,
            "unique_anchors_augmented": len(self.augmented_anchor_keys),
            "initial_positive_anchors": self.initial_positive_anchors,
            "c_eff": self.get_effective_c(),
            "history": list(self.history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_round_budget(self) -> Tuple[int, Optional[int]]:
        """Compute (this_round_budget, remaining_total_budget)."""
        max_total: Optional[int] = None

        if self.max_aug_examples is not None:
            max_total = max(0, int(self.max_aug_examples))
        elif self.target_c_eff is not None and self.initial_positive_anchors:
            max_total = max(1, math.floor(self.target_c_eff * self.initial_positive_anchors))
        elif self.max_rounds is not None and self.default_num_aug > 0:
            max_total = self.max_rounds * self.default_num_aug

        remaining: Optional[int] = (
            max(0, max_total - self.total_augmented) if max_total is not None else None
        )
        if remaining is not None and remaining <= 0:
            return 0, 0

        if self.target_c_eff is not None and max_total is not None and self.max_rounds:
            per_round = math.ceil(remaining / (self.max_rounds - self.round_idx))
        else:
            per_round = self.default_num_aug if self.default_num_aug > 0 else (remaining or 0)

        if remaining is not None:
            per_round = min(per_round, remaining)

        return per_round, remaining

    def _filter_candidates(
        self, selected: Iterable[Any], limit: int
    ) -> Tuple[List[Any], List[str]]:
        """Filter out previously augmented anchors and within-batch duplicates."""
        filtered: List[Any] = []
        keys: List[str] = []
        seen_in_batch: Set[str] = set()

        for ex in selected:
            key = self._example_key(ex)
            if not self.allow_repeat_anchors and key in self.augmented_anchor_keys:
                continue
            if key in seen_in_batch:
                continue
            seen_in_batch.add(key)
            filtered.append(ex)
            keys.append(key)
            if len(filtered) >= limit:
                break

        return filtered, keys

    def _record_round(
        self,
        epoch: Optional[int],
        selected_count: int,
        fresh_candidates: int,
        requested: int,
        added: int,
    ) -> None:
        self.history.append(
            {
                "round": self.round_idx,
                "epoch": epoch,
                "selected_candidates": selected_count,
                "fresh_candidates": fresh_candidates,
                "budget_requested": requested,
                "augmented_added": added,
                "total_augmented": self.total_augmented,
                "c_eff": self.get_effective_c(),
            }
        )

    @staticmethod
    def _example_key(example: Any) -> str:
        """Build a stable key for deduplicating anchors regardless of schema."""
        if isinstance(example, dict):
            for id_field in ("pair_id", "id", "index"):
                if id_field in example and example[id_field] is not None:
                    return f"{id_field}:{example[id_field]}"
            if "ltable_id" in example and "rtable_id" in example:
                return f"pair:{example['ltable_id']}:{example['rtable_id']}"
            payload = json.dumps(example, sort_keys=True, default=str)
        else:
            for attr in ("pair_id", "id"):
                val = getattr(example, attr, None)
                if val is not None:
                    return f"{attr}:{val}"
            payload = repr(example)

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _count_positive_anchors(train_dataset: Any) -> Optional[int]:
        """Best-effort count of seed positive matches (N_a) without triggering tokenization."""
        if train_dataset is None:
            return None

        explicit = getattr(train_dataset, "num_positives", None)
        if isinstance(explicit, int) and explicit > 0:
            return explicit

        # Check common label container attributes
        for attr in ("labels", "y"):
            labels = getattr(train_dataset, attr, None)
            if labels is not None:
                try:
                    return sum(1 for lbl in labels if int(lbl) == 1)
                except (TypeError, ValueError):
                    pass

        # Check DataFrame-backed datasets
        df = getattr(train_dataset, "df", None)
        if df is not None and hasattr(df, "__getitem__") and "label" in df:
            try:
                return int((df["label"] == 1).sum())
            except Exception:
                pass

        # Check raw pair/example lists of dicts or tuples
        for attr in ("examples", "pairs", "data"):
            items = getattr(train_dataset, attr, None)
            if isinstance(items, (list, tuple)) and items:
                count = 0
                matched_schema = False
                for item in items:
                    lbl = None
                    if isinstance(item, dict) and "label" in item:
                        lbl = item["label"]
                        matched_schema = True
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        lbl = item[-1]
                        matched_schema = True
                    if lbl is not None:
                        try:
                            if int(lbl) == 1:
                                count += 1
                        except (TypeError, ValueError):
                            pass
                if matched_schema:
                    return count

        return None
