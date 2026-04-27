"""Tests for src.common.metrics."""

from src.common.metrics import ordering_violations, plan_accuracy


class TestOrderingViolations:
    def test_identical_plans(self) -> None:
        sigs = ["move r a b", "pickup r p b", "drop r p c"]
        assert ordering_violations(sigs, sigs) == 0

    def test_reversed_plan(self) -> None:
        gold = ["A", "B", "C"]
        predicted = ["C", "B", "A"]
        # C is at pos 2, B at 1 (violation: 1 < 2), A at 0 (violation: 0 < 1)
        assert ordering_violations(gold, predicted) == 2

    def test_single_swap(self) -> None:
        gold = ["A", "B", "C"]
        predicted = ["B", "A", "C"]
        # B at pos 1, A at pos 0 (violation: 0 < 1), C at pos 2 (ok)
        assert ordering_violations(gold, predicted) == 1

    def test_extra_actions_ignored(self) -> None:
        gold = ["A", "B"]
        predicted = ["A", "X", "B"]
        # X not in gold — skipped. A at 0, B at 1 — monotone.
        assert ordering_violations(gold, predicted) == 0

    def test_empty_predicted(self) -> None:
        assert ordering_violations(["A", "B"], []) == 0

    def test_empty_gold(self) -> None:
        assert ordering_violations([], ["A", "B"]) == 0

    def test_both_empty(self) -> None:
        assert ordering_violations([], []) == 0

    def test_no_overlap(self) -> None:
        assert ordering_violations(["A", "B"], ["X", "Y"]) == 0


class TestPlanAccuracy:
    def test_perfect_match(self) -> None:
        sigs = ["A", "B", "C"]
        assert plan_accuracy(sigs, sigs) == 1.0

    def test_no_match(self) -> None:
        assert plan_accuracy(["A", "B"], ["X", "Y"]) == 0.0

    def test_partial_match(self) -> None:
        # A matches, B matches, X doesn't, D matches -> 3/4
        assert plan_accuracy(["A", "B", "C", "D"], ["A", "B", "X", "D"]) == 0.75

    def test_shorter_predicted(self) -> None:
        # Only first 2 compared, both match: 2/3
        acc = plan_accuracy(["A", "B", "C"], ["A", "B"])
        assert abs(acc - 2 / 3) < 1e-9

    def test_longer_predicted(self) -> None:
        # Zip truncates to gold length: 2/2 = 1.0
        assert plan_accuracy(["A", "B"], ["A", "B", "C"]) == 1.0

    def test_empty_gold_empty_predicted(self) -> None:
        assert plan_accuracy([], []) == 1.0

    def test_empty_gold_nonempty_predicted(self) -> None:
        assert plan_accuracy([], ["A"]) == 0.0

    def test_nonempty_gold_empty_predicted(self) -> None:
        assert plan_accuracy(["A", "B"], []) == 0.0
