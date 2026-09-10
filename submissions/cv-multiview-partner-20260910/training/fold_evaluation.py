"""依据固定验证分区汇总预测；无真值时只报告覆盖，不推造准确率。"""
from fold_protocol import validate_fold_plan


def _predictions(value, allowed):
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, dict) and set(item) == {"sample_id", "prediction"}:
                pairs.append((item["sample_id"], item["prediction"]))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append(tuple(item))
            else:
                raise ValueError("预测列表须包含身份与类型二元组，或 sample_id/prediction 对象")
    else:
        raise ValueError("预测必须是内存中的映射或身份列表；不接受路径")
    result = {}
    for sid, label in pairs:
        if not isinstance(sid, str) or sid not in allowed:
            raise ValueError("预测含未知样本身份")
        if sid in result:
            raise ValueError("预测含重复样本身份：" + sid)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("预测类型须为非空字符串")
        result[sid] = label
    return result


def _summary(rows, predictions, baseline):
    ids = [r["sample_id"] for r in rows]
    truth = {r["sample_id"]: set(r["labels"]) for r in rows if r["labels"]}
    present = [sid for sid in ids if sid in predictions]
    labelled_present = [sid for sid in present if sid in truth]
    missing = sorted(set(ids) - set(present))
    full = bool(ids) and not missing and len(truth) == len(ids)
    hits = sum(predictions[sid] in truth[sid] for sid in labelled_present)
    result = {"sample_count": len(ids), "prediction_count": len(present),
              "coverage": len(present) / len(ids) if ids else None,
              "missing_ids": missing, "complete_predictions": not missing,
              "ground_truth_count": len(truth),
              "accuracy": hits / len(ids) if full else None, "hits": hits if full else None,
              "metric_scope": "all_samples" if full else "labelled_subset" if truth else "unavailable",
              "reason": None if full else "真值缺失" if not truth else "真值或预测不完整，仅报告有真值且已预测子集",
              "labelled_subset": {"ground_truth_count": len(truth), "prediction_count": len(labelled_present),
                                   "prediction_coverage": len(labelled_present) / len(truth) if truth else None,
                                   "coverage_of_all_samples": len(labelled_present) / len(ids) if ids else None,
                                   "hits": hits if labelled_present else None,
                                   "accuracy": hits / len(labelled_present) if labelled_present else None}}
    result["comparison"] = None
    result["positive_gain"] = None
    if baseline is not None:
        paired = [sid for sid in ids if sid in predictions and sid in baseline]
        paired_truth = [sid for sid in paired if sid in truth]
        gained = sum(predictions[sid] in truth[sid] and baseline[sid] not in truth[sid] for sid in paired_truth)
        lost = sum(predictions[sid] not in truth[sid] and baseline[sid] in truth[sid] for sid in paired_truth)
        comparison_full = full and len(paired) == len(ids)
        subset = {"sample_count": len(paired_truth), "sample_ids": sorted(paired_truth),
                  "coverage_of_all_samples": len(paired_truth) / len(ids) if ids else None,
                  "gained": gained if paired_truth else None, "lost": lost if paired_truth else None,
                  "net": gained - lost if paired_truth else None}
        result["comparison"] = {
            "paired_prediction_count": len(paired),
            "paired_coverage": len(paired) / len(ids) if ids else None,
            "baseline_missing_ids": sorted(sid for sid in ids if sid not in baseline),
            "candidate_missing_ids": missing,
            "scope": "all_samples" if comparison_full else "paired_labelled_subset" if paired_truth else "unavailable",
            "gained": gained if comparison_full else None,
            "lost": lost if comparison_full else None,
            "net": gained - lost if comparison_full else None,
            "reason": None if comparison_full else "真值缺失" if not truth else "基线、候选或真值覆盖不完整，不能外推全体增益",
            "paired_labelled_subset": subset,
        }
        result["positive_gain"] = gained > lost if comparison_full else None
    return result


def summarize_predictions(plan, predictions, baseline_predictions=None):
    """预测仅表示已登记类别字符串；开放标签词表由调用者的上游输出合同校验。

    本模块不会把输入字符串当成路径读取，也不会用候选一致率构造标签。
    所有分区均固定来自完整计划，缺失预测保留在覆盖率分母。
    """
    validate_fold_plan(plan)
    allowed = set(plan["assignments"])
    candidate = _predictions(predictions, allowed)
    baseline = _predictions(baseline_predictions, allowed) if baseline_predictions is not None else None
    rows = plan["rows"]
    result = {"schema": "c4-fold-evaluation-v2", "plan_sha256": plan["plan_sha256"],
              "mode": plan["mode"], "n_splits": plan["n_splits"],
              "overall": _summary(rows, candidate, baseline), "folds": [], "categories": {}}
    for fold in range(plan["n_splits"]):
        subset = [r for r in rows if plan["assignments"][r["sample_id"]] == fold]
        result["folds"].append({"fold": fold, **_summary(subset, candidate, baseline)})
    for category in sorted({r["category"] for r in rows}):
        result["categories"][category] = _summary([r for r in rows if r["category"] == category], candidate, baseline)
    groups = {}
    for row in rows:
        groups.setdefault(row["image_sha256"], []).append(row)
    representatives = [min(group, key=lambda r: r["sample_id"]) for group in groups.values()]
    representatives.sort(key=lambda r: r["sample_id"])
    def inconsistent(prediction_map):
        if prediction_map is None:
            return None
        return sum(len({prediction_map[r["sample_id"]] for r in group if r["sample_id"] in prediction_map}) > 1
                   for group in groups.values())
    conflicting_truth = sum(len({tuple(r["labels"]) for r in group if r["labels"]}) > 1 for group in groups.values())
    unique = {"representative_strategy": "每个 image_sha256 组固定取 sample_id 字典序最小者；不依据预测或标签选择，不用其他副本补代表缺失",
              "unique_count": len(groups), "duplicate_copies": len(rows) - len(groups),
              "representative_ids": [r["sample_id"] for r in representatives],
              "inconsistent_prediction_group_count": inconsistent(candidate),
              "baseline_inconsistent_prediction_group_count": inconsistent(baseline),
              "conflicting_ground_truth_group_count": conflicting_truth,
              "overall": _summary(representatives, candidate, baseline), "folds": [], "categories": {}}
    for fold in range(plan["n_splits"]):
        subset = [r for r in representatives if plan["assignments"][r["sample_id"]] == fold]
        unique["folds"].append({"fold": fold, **_summary(subset, candidate, baseline)})
    for category in sorted({r["category"] for r in representatives}):
        unique["categories"][category] = _summary([r for r in representatives if r["category"] == category], candidate, baseline)
    result["primary_quality_scope"] = "unique_content"
    result["file_level_scope"] = "overall/folds/categories 保留原文件口径；副本可能重复计权"
    result["unique_content"] = unique
    return result
