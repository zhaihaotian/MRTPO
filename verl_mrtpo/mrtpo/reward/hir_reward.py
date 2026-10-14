"""Per-criterion HIR reward entrypoint for verl."""

import copy
import json
import logging

LOGGER = logging.getLogger(__name__)
_logged_checker_error = False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return the mean rule score and ordered binary criterion scores."""
    global _logged_checker_error
    truth = None
    try:
        truth = json.loads(ground_truth) if isinstance(ground_truth, str) else copy.deepcopy(ground_truth)
        instruction_ids = truth["instruction_id_list"]
        kwargs = truth["kwargs"]
        if len(instruction_ids) != len(kwargs) or not instruction_ids:
            raise ValueError("instruction_id_list and kwargs must have equal nonzero length")
        if extra_info is not None and "n_criteria" in extra_info:
            if len(instruction_ids) != int(extra_info["n_criteria"]):
                raise ValueError("n_criteria differs from instruction_id_list")
        source = truth["source"]
        if source == "type1":
            from mrtpo.reward.if_score.ifeval.evaluation_lib import test_ifeval_strict as checker
        elif source == "type2":
            from mrtpo.reward.if_score.ifbench.evaluation_lib import test_ifbench_strict as checker
        else:
            raise ValueError(f"unsupported source: {source}")
        scores = []
        errors = 0
        for instruction_id, keyword_args in zip(instruction_ids, kwargs):
            single = {"instruction_id_list": [instruction_id], "kwargs": [copy.deepcopy(keyword_args)]}
            try:
                _, _, judged = checker(solution_str, single)
                if len(judged) != 1:
                    raise ValueError("checker returned wrong criterion count")
                scores.append(float(judged[0]))
            except Exception:
                errors += 1
                scores.append(0.0)
                if not _logged_checker_error:
                    LOGGER.exception("HIR criterion checker failed; subsequent failures are counted")
                    _logged_checker_error = True
        return {
            "score": sum(scores) / len(scores), "all_pass": float(all(scores)),
            "n_criteria": len(scores), "criteria_scores": json.dumps(scores),
            "checker_errors": errors,
        }
    except Exception:
        if not _logged_checker_error:
            LOGGER.exception("HIR reward input failed; subsequent failures are counted")
            _logged_checker_error = True
        count = 0
        try:
            if isinstance(extra_info, dict) and "n_criteria" in extra_info:
                count = max(0, int(extra_info["n_criteria"]))
            elif isinstance(truth, dict):
                count = len(truth.get("instruction_id_list", []))
        except (TypeError, ValueError):
            pass
        return {"score": 0.0, "all_pass": 0.0, "n_criteria": count,
                "criteria_scores": json.dumps([0.0] * count), "checker_errors": max(count, 1)}
