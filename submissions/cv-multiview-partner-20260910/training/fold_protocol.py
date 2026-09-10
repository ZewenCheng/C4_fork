"""可配置分组多标签折协议；仅声明可核验的独立性，不执行训练。"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "c4-fold-protocol-v2"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _normalise(rows, allow_empty_labels=False):
    result = {}
    for row in rows:
        required = ("sample_id", "image_sha256", "group_id", "category")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError("样本身份、图像摘要、分组与类别必须明确；未知分组须显式生成去重组")
        labels = row.get("labels")
        if not isinstance(labels, list) or (not labels and not allow_empty_labels) or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("标签必须是字符串列表；拟合交叉验证不允许空标签")
        value = {k: row[k] for k in required}
        value["labels"] = sorted(set(labels))
        sid = value["sample_id"]
        if sid in result and result[sid] != value:
            raise ValueError("重复 sample_id 内容冲突：" + sid)
        result[sid] = value
    if not result:
        raise ValueError("数据为空")
    return [result[sid] for sid in sorted(result)]


def _components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    known = {}
    for i, row in enumerate(rows):
        for field in ("group_id", "image_sha256"):
            key = (field, row[field])
            if key in known:
                a, b = find(i), find(known[key])
                parent[max(a, b)] = min(a, b)
            known[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def build_fold_plan(rows, n_splits=5, seed=20260910, group_provenance="unknown",
                    mode="evaluation_only", training_authorized=False):
    """确定性贪心分层；同采集组或相同字节图片的传递闭包不可拆分。

    group_provenance 仅接受 unknown / verified-acquisition-groups。
    后者必须由调用者持有采集组证据；本函数无法从名称证明采集独立。
    默认只生成评估分区；补集不授权拟合。训练交叉验证必须显式开启并提供授权。
    """
    if type(n_splits) is not int or n_splits < 2 or type(seed) is not int:
        raise ValueError("折数须为至少 2 的整数，种子须为整数")
    if group_provenance not in ("unknown", "verified-acquisition-groups"):
        raise ValueError("不支持的分组来源声明")
    if mode not in ("evaluation_only", "cross_validation"):
        raise ValueError("模式必须明确区分验证分区与拟合交叉验证")
    if type(training_authorized) is not bool or (mode == "cross_validation" and not training_authorized):
        raise ValueError("拟合交叉验证必须具有明确的数据训练授权")
    if mode == "evaluation_only" and training_authorized:
        raise ValueError("仅评估模式不能携带训练授权声明")
    rows = _normalise(rows, allow_empty_labels=(mode == "evaluation_only"))
    groups = _components(rows)
    if len(groups) < n_splits:
        raise ValueError("合并重复图像与采集组后，独立组数少于折数")
    def features(group):
        counts = Counter()
        for row in group:
            counts[("category", row["category"])] += 1
            for label in row["labels"]:
                counts[("label", row["category"], label)] += 1
        return counts
    counts = [features(group) for group in groups]
    total = sum(counts, Counter())
    order = sorted(range(len(groups)), key=lambda i: (
        -sum(v / total[k] for k, v in counts[i].items()), -len(groups[i]),
        digest([seed, [r["sample_id"] for r in groups[i]]])))
    bins = [Counter() for _ in range(n_splits)]
    sizes = [0] * n_splits
    assignment, component_ids = {}, {}
    for position, i in enumerate(order):
        group, count = groups[i], counts[i]
        # 比较全局平方偏差的增量；稀有标签优先，样本数量单独平衡。
        def cost(fold):
            value = sum(((bins[fold][k] + v) ** 2 - bins[fold][k] ** 2) /
                        max(total[k] / n_splits, 1) ** 2 for k, v in count.items())
            value += ((sizes[fold] + len(group)) ** 2 - sizes[fold] ** 2) / (len(rows) / n_splits) ** 2
            return value, sizes[fold], digest([seed, i, fold])
        choices = [j for j in range(n_splits) if sizes[j] == 0] if position < n_splits else range(n_splits)
        chosen = min(choices, key=cost)
        bins[chosen].update(count)
        sizes[chosen] += len(group)
        cid = digest([r["sample_id"] for r in group])
        for row in group:
            assignment[row["sample_id"]] = chosen
            component_ids[row["sample_id"]] = cid
    all_ids = sorted(assignment)
    folds = []
    for fold in range(n_splits):
        eval_ids = [sid for sid in all_ids if assignment[sid] == fold]
        complement_ids = [sid for sid in all_ids if assignment[sid] != fold]
        train_ids = complement_ids if mode == "cross_validation" else []
        subset = [row for row in rows if assignment[row["sample_id"]] == fold]
        folds.append({"fold": fold, "train_ids": train_ids, "eval_ids": eval_ids,
                      "complement_ids": complement_ids, "complement_ids_sha256": digest(complement_ids),
                      "train_ids_sha256": digest(train_ids), "eval_ids_sha256": digest(eval_ids),
                      "eval_size": len(eval_ids), "eval_group_count": len({component_ids[sid] for sid in eval_ids}),
                      "eval_category_counts": dict(sorted(Counter(r["category"] for r in subset).items())),
                      "eval_label_counts": dict(sorted(Counter(label for r in subset for label in r["labels"]).items()))})
    plan = {"schema": SCHEMA, "n_splits": n_splits, "seed": seed, "rows": rows,
            "mode": mode, "training_authorized": training_authorized,
            "mechanism": "validation_partitions" if mode == "evaluation_only" else "cross_validation",
            "dataset_sha256": digest(rows), "group_provenance": group_provenance,
            "independence": "group-independent" if group_provenance == "verified-acquisition-groups" else "duplicate-grouped",
            "independence_limit": "采集来源独立性依赖外部证据；去重不证明采集独立，稀有标签可能缺折",
            "assignments": dict(sorted(assignment.items())), "component_ids": component_ids, "folds": folds}
    labelled = sum(bool(row["labels"]) for row in rows)
    plan["has_ground_truth"] = labelled == len(rows)
    plan["ground_truth_sample_count"] = labelled
    plan["ground_truth_metrics"] = {"value": None, "reason": "尚未输入预测进行计算" if labelled == len(rows) else "真值缺失或不完整，不能计算全体准确率"}
    plan["plan_sha256"] = digest(plan)
    validate_fold_plan(plan)
    return plan


def validate_fold_plan(plan):
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("schema") != SCHEMA or digest(body) != plan.get("plan_sha256"):
        raise ValueError("折计划内容或版本被修改")
    rows = _normalise(plan["rows"], allow_empty_labels=(plan.get("mode") == "evaluation_only"))
    if digest(rows) != plan["dataset_sha256"]:
        raise ValueError("数据摘要不一致")
    labelled = sum(bool(row["labels"]) for row in rows)
    if plan.get("has_ground_truth") is not (labelled == len(rows)) or plan.get("ground_truth_sample_count") != labelled:
        raise ValueError("真值状态与标签内容不一致")
    assignments = plan["assignments"]
    all_ids = {r["sample_id"] for r in rows}
    n = plan["n_splits"]
    mode = plan.get("mode")
    if (mode not in ("evaluation_only", "cross_validation")
            or plan.get("training_authorized") is not (mode == "cross_validation")
            or plan.get("mechanism") != ("validation_partitions" if mode == "evaluation_only" else "cross_validation")):
        raise ValueError("评估模式、训练授权与机制声明不一致")
    expected_independence = {"unknown": "duplicate-grouped", "verified-acquisition-groups": "group-independent"}
    if (type(n) is not int or n < 2 or plan.get("group_provenance") not in expected_independence
            or plan.get("independence") != expected_independence[plan["group_provenance"]]):
        raise ValueError("折数或独立性声明不一致")
    if set(assignments) != all_ids or set(assignments.values()) != set(range(n)):
        raise ValueError("折覆盖不完整")
    for group in _components(rows):
        if len({assignments[r["sample_id"]] for r in group}) != 1:
            raise ValueError("采集组或重复图片跨折泄漏")
        cid = digest([r["sample_id"] for r in group])
        if any(plan["component_ids"].get(r["sample_id"]) != cid for r in group):
            raise ValueError("连通分组身份不一致")
    if len(plan["folds"]) != n:
        raise ValueError("折列表不完整")
    for i, fold in enumerate(plan["folds"]):
        expected_eval = sorted(sid for sid in all_ids if assignments[sid] == i)
        expected_complement = sorted(all_ids - set(expected_eval))
        expected_train = expected_complement if mode == "cross_validation" else []
        if fold.get("complement_ids") != expected_complement or fold.get("complement_ids_sha256") != digest(expected_complement):
            raise ValueError("验证分区补集身份不匹配；补集不自动具有训练授权")
        if fold["fold"] != i or fold["eval_ids"] != expected_eval or fold["train_ids"] != expected_train:
            raise ValueError("训练与验证身份不匹配")
        if fold["eval_ids_sha256"] != digest(expected_eval) or fold["train_ids_sha256"] != digest(expected_train):
            raise ValueError("折身份摘要不匹配")
    return True


def validate_training_sources(plan, fold, training_ids, calibration_ids=(), checkpoints=()):
    """核验所有数据驱动拟合步骤；公开预训练记录未知来源而不伪称可证无重叠。

    checkpoints: {name, kind: public-pretrained|task-fitted, training_ids: list|None}。
    cross_validation 的任务资产须使用该折训练身份；evaluation_only 只能复用
    明确未训练本测试池任一样本的外部资产，不得对测试池拟合或校准。
    """
    validate_fold_plan(plan)
    if type(fold) is not int or not 0 <= fold < plan["n_splits"]:
        raise ValueError("折编号无效")
    allowed = set(plan["folds"][fold]["train_ids"])
    evaluation = set(plan["folds"][fold]["eval_ids"])
    def check(ids, name):
        ids = list(ids)
        if any(not isinstance(sid, str) for sid in ids) or len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError(name + " 包含验证折、未知或重复身份")
        return sorted(ids)
    training = check(training_ids, "训练")
    if not training and plan["mode"] == "cross_validation":
        raise ValueError("训练身份为空")
    calibration = check(calibration_ids, "校准")
    audit = []
    for checkpoint in checkpoints:
        kind, ids = checkpoint.get("kind"), checkpoint.get("training_ids")
        if kind not in ("task-fitted", "public-pretrained") or not checkpoint.get("name"):
            raise ValueError("检查点来源声明缺失")
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(sid, str) or not sid for sid in ids)
                                or len(ids) != len(set(ids))):
            raise ValueError("检查点训练身份必须是唯一非空字符串列表")
        forbidden = set(plan["assignments"]) if plan["mode"] == "evaluation_only" else evaluation
        if ids is not None and forbidden.intersection(ids):
            raise ValueError("检查点曾使用当前验证折")
        if kind == "task-fitted":
            if not ids:
                raise ValueError("任务拟合检查点缺少训练身份")
            if plan["mode"] == "cross_validation":
                check(ids, "任务检查点")
        audit.append({"name": checkpoint["name"], "kind": kind,
                      "training_ids_sha256": digest(sorted(ids)) if ids is not None else None,
                      "overlap_status": "unknown-public-pretraining" if ids is None else "declared-ids-checked"})
    return {"fold": fold, "plan_sha256": plan["plan_sha256"], "training_ids_sha256": digest(training),
            "calibration_ids_sha256": digest(calibration), "checkpoints": audit,
            "calibration_is_disjoint_from_model_fit": not bool(set(training) & set(calibration))}


def cache_identity(plan, fold, training_assets, preprocessing, output_schema):
    validate_fold_plan(plan)
    if type(fold) is not int or not 0 <= fold < plan["n_splits"]:
        raise ValueError("折编号无效")
    if not training_assets or not preprocessing or not output_schema:
        raise ValueError("训练资产、预处理及输出合同必须明确")
    payload = {"schema": SCHEMA, "dataset_sha256": plan["dataset_sha256"],
               "plan_sha256": plan["plan_sha256"], "fold": fold,
               "training_assets": training_assets, "preprocessing": preprocessing, "output_schema": output_schema}
    return {"key": digest(payload), "payload": payload}


def write_receipt(path, identity, result):
    """仅适用于单作业拥有的路径；调用者必须使用含折号及 identity 的目录。"""
    path = Path(path)
    body = {"identity": identity, "result": result, "result_sha256": digest(result)}
    body["receipt_sha256"] = digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_receipt(path, identity)
        if existing != result:
            raise ValueError("已有完成凭证不可覆盖为不同结果")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_receipt(path, identity):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("identity") != identity or digest(body) != receipt.get("receipt_sha256"):
        raise ValueError("恢复凭证跨折、版本不符或已损坏")
    if digest(receipt["result"]) != receipt["result_sha256"]:
        raise ValueError("恢复结果摘要不匹配")
    return receipt["result"]
