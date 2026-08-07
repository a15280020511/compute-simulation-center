#!/usr/bin/env python3
"""Deterministic NetworkX orchestration for allowlisted serial compute pipelines."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from jsonschema import Draft202012Validator

from operation_validation import validate_operation_inputs
from pipeline_adapters import ADAPTERS, PipelineAdapterError

HERE = Path(__file__).resolve().parent
PIPELINE_REGISTRY_PATH = HERE / "pipeline-registry.json"
CONTRACT_REGISTRY_PATH = HERE / "operation-contract-registry.json"
MAX_STAGES = 8


class PipelineEngineError(ValueError):
    """Raised when a registered pipeline is invalid or a stage cannot execute safely."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PipelineEngineError(f"JSON root must be an object: {path.name}")
    return value


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_contracts() -> dict[str, Mapping[str, Any]]:
    document = _load(CONTRACT_REGISTRY_PATH)
    if document.get("schema_version") != "compute-operation-contract-registry-v1":
        raise PipelineEngineError("invalid operation contract registry schema")
    if document.get("status") != "controlled-preview":
        raise PipelineEngineError("operation contracts must remain controlled-preview")
    raw = document.get("contracts")
    if not isinstance(raw, Mapping) or not raw:
        raise PipelineEngineError("operation contract registry is empty")
    contracts: dict[str, Mapping[str, Any]] = {}
    for operation, value in raw.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("output_schema"), Mapping):
            raise PipelineEngineError(f"invalid operation contract: {operation}")
        Draft202012Validator.check_schema(dict(value["output_schema"]))
        contracts[str(operation)] = value
    return contracts


def validate_registry(document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(document or _load(PIPELINE_REGISTRY_PATH))
    if value.get("schema_version") != "compute-pipeline-registry-v1":
        raise PipelineEngineError("invalid pipeline registry schema")
    required_policy = {
        "status": "controlled-preview",
        "engine": "networkx",
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "dynamic_stage_injection_allowed": False,
        "ticket_supplied_adapter_code_allowed": False,
        "cycles_allowed": False,
    }
    for key, expected in required_policy.items():
        if value.get(key) != expected:
            raise PipelineEngineError(f"unsafe pipeline registry policy: {key}")
    if int(value.get("maximum_stages") or 0) != MAX_STAGES:
        raise PipelineEngineError(f"maximum_stages must equal {MAX_STAGES}")

    contracts = load_contracts()
    pipelines = value.get("pipelines")
    if not isinstance(pipelines, list) or not pipelines:
        raise PipelineEngineError("pipeline registry must contain at least one pipeline")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in pipelines:
        if not isinstance(raw, Mapping):
            raise PipelineEngineError("pipeline entry must be an object")
        pipeline = dict(raw)
        pipeline_id = str(pipeline.get("id") or "")
        stages = pipeline.get("stages")
        if not pipeline_id or pipeline_id in by_id:
            raise PipelineEngineError(f"invalid or duplicate pipeline id: {pipeline_id!r}")
        if pipeline.get("maturity") != "controlled-preview":
            raise PipelineEngineError(f"pipeline must remain controlled-preview: {pipeline_id}")
        if pipeline.get("resume_policy") != "hash-gated-manual-resubmit":
            raise PipelineEngineError(f"unsupported resume policy: {pipeline_id}")
        if not isinstance(stages, list) or not 2 <= len(stages) <= MAX_STAGES:
            raise PipelineEngineError(f"pipeline must contain 2 to {MAX_STAGES} stages: {pipeline_id}")

        graph = nx.DiGraph()
        order_index: dict[str, int] = {}
        stage_map: dict[str, dict[str, Any]] = {}
        for index, stage_raw in enumerate(stages):
            if not isinstance(stage_raw, Mapping):
                raise PipelineEngineError(f"invalid stage in {pipeline_id}")
            stage = dict(stage_raw)
            stage_id = str(stage.get("id") or "")
            operation = str(stage.get("operation") or "")
            adapter = str(stage.get("adapter") or "")
            dependencies = stage.get("depends_on")
            if not stage_id or stage_id in stage_map:
                raise PipelineEngineError(f"invalid or duplicate stage id in {pipeline_id}: {stage_id!r}")
            if operation not in contracts:
                raise PipelineEngineError(f"pipeline operation lacks output contract: {operation}")
            if adapter not in ADAPTERS:
                raise PipelineEngineError(f"pipeline stage uses unknown adapter: {adapter}")
            if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
                raise PipelineEngineError(f"invalid dependencies for stage {stage_id}")
            graph.add_node(stage_id)
            stage_map[stage_id] = stage
            order_index[stage_id] = index
        for stage_id, stage in stage_map.items():
            for dependency in stage["depends_on"]:
                if dependency not in stage_map or dependency == stage_id:
                    raise PipelineEngineError(f"unknown or self dependency for stage {stage_id}: {dependency}")
                graph.add_edge(str(dependency), stage_id)
        if not nx.is_directed_acyclic_graph(graph):
            raise PipelineEngineError(f"pipeline contains a cycle: {pipeline_id}")
        ordered = list(nx.lexicographical_topological_sort(graph, key=lambda node: order_index[str(node)]))
        if len(graph.edges) != len(stages) - 1:
            raise PipelineEngineError(f"pipeline must be a single serial chain: {pipeline_id}")
        for index, stage_id in enumerate(ordered):
            indegree = graph.in_degree(stage_id)
            outdegree = graph.out_degree(stage_id)
            expected_in = 0 if index == 0 else 1
            expected_out = 0 if index == len(ordered) - 1 else 1
            if indegree != expected_in or outdegree != expected_out:
                raise PipelineEngineError(f"pipeline branching or disconnected stages are forbidden: {pipeline_id}")
        first_operation = str(stage_map[ordered[0]]["operation"])
        if pipeline.get("entry_operation") != first_operation:
            raise PipelineEngineError(f"entry_operation mismatch: {pipeline_id}")
        if pipeline.get("result_stage") != ordered[-1]:
            raise PipelineEngineError(f"result_stage must be the final serial stage: {pipeline_id}")
        pipeline["stage_order"] = ordered
        pipeline["stage_map"] = stage_map
        by_id[pipeline_id] = pipeline
    return {
        "schema_version": value["schema_version"],
        "engine": "networkx",
        "network_policy": "deny",
        "automatic_parallel_execution": False,
        "pipelines": by_id,
    }


def resolve_pipeline_ticket(ticket: Mapping[str, Any]) -> dict[str, Any] | None:
    pipeline_ref = ticket.get("pipeline")
    if not isinstance(pipeline_ref, Mapping) or str(pipeline_ref.get("stage_id") or "") != "pipeline":
        return None
    registry = validate_registry()
    pipeline_id = str(pipeline_ref.get("pipeline_id") or "")
    pipeline = registry["pipelines"].get(pipeline_id)
    if not isinstance(pipeline, Mapping):
        raise PipelineEngineError(f"unregistered fixed pipeline requested: {pipeline_id}")
    if str(ticket.get("operation") or "") != str(pipeline.get("entry_operation") or ""):
        raise PipelineEngineError(
            f"ticket operation must equal pipeline entry_operation {pipeline.get('entry_operation')}"
        )
    return dict(pipeline)


def _validate_output(operation: str, result: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]) -> None:
    contract = contracts.get(operation)
    if not isinstance(contract, Mapping):
        raise PipelineEngineError(f"missing output contract for operation {operation}")
    validator = Draft202012Validator(dict(contract["output_schema"]))
    errors = sorted(validator.iter_errors(result), key=lambda item: list(item.absolute_path))
    if errors:
        rendered = []
        for error in errors[:10]:
            path = ".".join(str(item) for item in error.absolute_path) or "$"
            rendered.append(f"{path}: {error.message}")
        raise PipelineEngineError(f"stage output contract failed for {operation}: {'; '.join(rendered)}")


def run_pipeline_ticket(
    ticket: Mapping[str, Any],
    output_dir: Path,
    operations: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    pipeline = resolve_pipeline_ticket(ticket)
    if pipeline is None:
        raise PipelineEngineError("ticket does not request a fixed single-ticket pipeline")
    contracts = load_contracts()
    stage_order = list(pipeline["stage_order"])
    stage_map = dict(pipeline["stage_map"])
    missing_handlers = sorted(
        {str(stage_map[stage_id]["operation"]) for stage_id in stage_order} - set(operations)
    )
    if missing_handlers:
        raise PipelineEngineError("pipeline handlers are unavailable: " + ", ".join(missing_handlers))

    initial_inputs = ticket.get("inputs")
    if not isinstance(initial_inputs, Mapping):
        raise PipelineEngineError("pipeline ticket inputs must be an object")
    state = {
        "schema_version": "compute-pipeline-state-v1",
        "pipeline_id": pipeline["id"],
        "status": "RUNNING",
        "engine": "networkx",
        "execution_policy": "strict-serial",
        "automatic_parallel_execution": False,
        "network_used": False,
        "model_calls": 0,
        "stages": [
            {"stage_id": stage_id, "operation": stage_map[stage_id]["operation"], "status": "PENDING"}
            for stage_id in stage_order
        ],
    }
    _write_json(output_dir / "compute-pipeline-state.json", state)
    stage_results: dict[str, dict[str, Any]] = {}
    stage_receipts: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
    stage_dir = output_dir / "pipeline-stages"
    started = time.perf_counter()

    try:
        for index, stage_id in enumerate(stage_order):
            stage = stage_map[stage_id]
            operation = str(stage["operation"])
            adapter_name = str(stage["adapter"])
            state["stages"][index]["status"] = "RUNNING"
            _write_json(output_dir / "compute-pipeline-state.json", state)
            try:
                stage_inputs = ADAPTERS[adapter_name](initial_inputs, stage_results, stage)
            except PipelineAdapterError as exc:
                raise PipelineEngineError(f"adapter {adapter_name} failed at {stage_id}: {exc}") from exc
            derived_ticket = dict(ticket)
            derived_ticket["operation"] = operation
            derived_ticket["inputs"] = stage_inputs
            validate_operation_inputs(derived_ticket)
            input_sha = _canonical_sha(stage_inputs)
            _write_json(stage_dir / f"{index + 1:02d}-{stage_id}-input.json", stage_inputs)
            stage_started = time.perf_counter()
            result = operations[operation](stage_inputs)
            stage_elapsed[stage_id] = round(time.perf_counter() - stage_started, 6)
            if not isinstance(result, Mapping):
                raise PipelineEngineError(f"stage {stage_id} returned a non-object result")
            result_dict = dict(result)
            _validate_output(operation, result_dict, contracts)
            output_sha = _canonical_sha(result_dict)
            stage_results[stage_id] = result_dict
            _write_json(stage_dir / f"{index + 1:02d}-{stage_id}-output.json", result_dict)
            receipt = {
                "stage_id": stage_id,
                "operation": operation,
                "adapter": adapter_name,
                "status": "PASS",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
            }
            stage_receipts.append(receipt)
            state["stages"][index].update(receipt)
            _write_json(output_dir / "compute-pipeline-state.json", state)
    except Exception:
        state["status"] = "FAILED"
        for row in state["stages"]:
            if row["status"] == "RUNNING":
                row["status"] = "FAILED"
        _write_json(output_dir / "compute-pipeline-state.json", state)
        raise

    elapsed = time.perf_counter() - started
    state["status"] = "PASS"
    state["pipeline_sha256"] = _canonical_sha(stage_receipts)
    _write_json(output_dir / "compute-pipeline-state.json", state)

    import numpy as np
    import scipy

    final_stage = str(pipeline["result_stage"])
    result_data = {
        "pipeline_id": pipeline["id"],
        "pipeline_maturity": pipeline["maturity"],
        "engine": "networkx",
        "execution_policy": "strict-serial",
        "automatic_parallel_execution": False,
        "stage_order": stage_order,
        "stage_receipts": stage_receipts,
        "stage_outputs": stage_results,
        "final_stage": final_stage,
        "final_result": stage_results[final_stage],
    }
    transfer = {
        "schema_version": "compute-result-v1",
        "task_id": str(ticket["task_id"]),
        "status": "success",
        "operation": str(ticket["operation"]),
        "objective": ticket.get("objective"),
        "input_sha256": _canonical_sha(ticket),
        "assumptions": ticket.get("assumptions", []),
        "evidence": ticket.get("evidence", []),
        "limitations": ticket.get("limitations", []),
        "results": result_data,
        "maturity_assessment": {
            "engineering_maturity": "controlled-preview",
            "evidence_maturity": "controlled-preview",
        },
        "software": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "execution": {
            "elapsed_seconds": round(elapsed, 6),
            "stage_elapsed_seconds": stage_elapsed,
            "network_used": False,
            "model_calls": 0,
            "reproducible": True,
            "automatic_parallel_execution": False,
        },
    }
    transfer["result_sha256"] = _canonical_sha(
        {
            "schema_version": transfer["schema_version"],
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "input_sha256": transfer["input_sha256"],
            "assumptions": transfer["assumptions"],
            "limitations": transfer["limitations"],
            "results": transfer["results"],
            "maturity_assessment": transfer["maturity_assessment"],
            "software": transfer["software"],
        }
    )
    _write_json(output_dir / "compute-result.json", transfer)
    _write_json(
        output_dir / "compute-audit.json",
        {
            "version": 1,
            "status": "PASS",
            "task_id": transfer["task_id"],
            "operation": transfer["operation"],
            "pipeline_id": pipeline["id"],
            "pipeline_sha256": state["pipeline_sha256"],
            "input_sha256": transfer["input_sha256"],
            "result_sha256": transfer["result_sha256"],
            "elapsed_seconds": transfer["execution"]["elapsed_seconds"],
            "model_calls": 0,
            "network_used": False,
            "automatic_parallel_execution": False,
            "secret_values_included": False,
        },
    )
    (output_dir / "compute-summary.md").write_text(
        "# COMPUTE_COMPLETED\n\n"
        f"- Task ID: `{transfer['task_id']}`\n"
        f"- Operation: `{transfer['operation']}`\n"
        f"- Pipeline: `{pipeline['id']}`\n"
        f"- Stage order: `{' -> '.join(stage_order)}`\n"
        f"- Result SHA256: `{transfer['result_sha256']}`\n"
        "- Orchestration engine: `networkx`\n"
        "- Execution policy: `strict-serial`\n"
        "- Automatic parallel execution: `false`\n"
        "- Model calls: `0`\n"
        "- Network used: `false`\n",
        encoding="utf-8",
    )
    return transfer


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the fixed NetworkX compute pipeline registry.")
    parser.add_argument("command", choices=["validate", "catalog"])
    args = parser.parse_args()
    registry = validate_registry()
    payload = {
        "status": "PASS",
        "engine": registry["engine"],
        "network_policy": registry["network_policy"],
        "automatic_parallel_execution": registry["automatic_parallel_execution"],
        "pipelines": {
            pipeline_id: {
                "entry_operation": row["entry_operation"],
                "stage_order": row["stage_order"],
                "maturity": row["maturity"],
            }
            for pipeline_id, row in registry["pipelines"].items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
