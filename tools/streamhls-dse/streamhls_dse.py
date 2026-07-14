#!/usr/bin/env python3
"""License-free, solver-neutral design-space optimizer for Stream-HLS.

The optimizer consumes and produces versioned JSON.  Keeping the solver
process outside the MLIR binary gives the compiler a stable boundary and
prevents solver-specific APIs from leaking into transformation passes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
RESOURCE_PRESSURE_SCALE = 1_000_000
LEXICOGRAPHIC_SCALE = RESOURCE_PRESSURE_SCALE + 1
INT64_SAFE_MAX = 2**62


class ProblemError(ValueError):
    """Raised when a DSE problem violates the versioned input contract."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, path: str, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise ProblemError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ProblemError(f"{path} must be at least {minimum}")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProblemError(f"{path} must be an array")
    return value


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProblemError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProblemError(f"{path} must be a non-empty string")
    return value


def _topological_order(
    node_ids: set[int], edges: list[dict[str, Any]]
) -> tuple[list[int], dict[int, list[str]], dict[int, list[str]]]:
    incoming = {node_id: [] for node_id in node_ids}
    outgoing = {node_id: [] for node_id in node_ids}
    successors = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target:
            raise ProblemError(f"edge {edge['id']} is a self-cycle")
        incoming[target].append(edge["id"])
        outgoing[source].append(edge["id"])
        successors[source].append(target)
        indegree[target] += 1

    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()

    if len(order) != len(node_ids):
        raise ProblemError("the DSE graph must be acyclic")
    return order, incoming, outgoing


def _validate_resources(
    resources: Any, path: str, resource_limits: dict[str, int]
) -> dict[str, int]:
    resources = _require_object(resources, path)
    for resource, amount in resources.items():
        if resource not in resource_limits:
            raise ProblemError(
                f"{path} uses resource {resource!r} without a declared limit"
            )
        _require_int(amount, f"{path}.{resource}", 0)
    return resources


def _validate_interface(interface: Any, path: str) -> dict[str, Any]:
    interface = _require_object(interface, path)
    _require_string(interface.get("class"), f"{path}.class")

    if "shape" in interface:
        shape = _require_list(interface["shape"], f"{path}.shape")
        for position, extent in enumerate(shape):
            _require_int(extent, f"{path}.shape[{position}]", 1)
    for name in ("layout", "token_order"):
        if name in interface:
            _require_string(interface[name], f"{path}.{name}")
    if "rate" in interface:
        rate = _require_object(interface["rate"], f"{path}.rate")
        _require_int(rate.get("tokens"), f"{path}.rate.tokens", 1)
        _require_int(rate.get("cycles"), f"{path}.rate.cycles", 1)
    return interface


def validate_problem(problem: Any) -> dict[str, Any]:
    """Validate and normalize a DSE problem without importing OR-Tools."""

    problem = _require_object(problem, "problem")
    if problem.get("schema_version") != SCHEMA_VERSION:
        raise ProblemError(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{problem.get('schema_version')!r}"
        )

    _require_int(problem.get("time_limit_seconds", 60), "time_limit_seconds", 1)
    _require_int(problem.get("random_seed", 0), "random_seed", 0)
    _require_int(problem.get("num_workers", 1), "num_workers", 1)

    objective = _require_object(problem.get("objective", {}), "objective")
    objective_mode = objective.get(
        "mode", "lexicographic_latency_resource_pressure"
    )
    if objective_mode not in (
        "latency",
        "lexicographic_latency_resource_pressure",
    ):
        raise ProblemError(
            "objective.mode must be 'latency' or "
            "'lexicographic_latency_resource_pressure'"
        )

    resource_limits = _require_object(
        problem.get("resource_limits", {}), "resource_limits"
    )
    for resource, limit in resource_limits.items():
        _require_string(resource, "resource limit name")
        _require_int(limit, f"resource_limits.{resource}", 0)

    nodes = _require_list(problem.get("nodes"), "nodes")
    if not nodes:
        raise ProblemError("nodes must not be empty")

    node_by_id: dict[int, dict[str, Any]] = {}
    candidate_index: dict[int, dict[int, int]] = {}
    for node_position, raw_node in enumerate(nodes):
        node_path = f"nodes[{node_position}]"
        node = _require_object(raw_node, node_path)
        node_id = _require_int(node.get("id"), f"{node_path}.id", 0)
        if node_id in node_by_id:
            raise ProblemError(f"duplicate node id {node_id}")
        node_by_id[node_id] = node

        candidates = _require_list(node.get("candidates"), f"{node_path}.candidates")
        if not candidates:
            raise ProblemError(f"node {node_id} has no candidates")
        candidate_index[node_id] = {}

        for candidate_position, raw_candidate in enumerate(candidates):
            candidate_path = f"node {node_id} candidate {candidate_position}"
            candidate = _require_object(raw_candidate, candidate_path)
            candidate_id = _require_int(candidate.get("id"), f"{candidate_path}.id", 0)
            if candidate_id in candidate_index[node_id]:
                raise ProblemError(
                    f"node {node_id} has duplicate candidate id {candidate_id}"
                )
            candidate_index[node_id][candidate_id] = candidate_position

            timing = _require_object(candidate.get("timing"), f"{candidate_path}.timing")
            first_output = _require_int(
                timing.get("first_output"),
                f"{candidate_path}.timing.first_output",
                0,
            )
            last_output = _require_int(
                timing.get("last_output"),
                f"{candidate_path}.timing.last_output",
                0,
            )
            if first_output > last_output:
                raise ProblemError(
                    f"{candidate_path}.timing.first_output must not exceed last_output"
                )
            _require_int(
                timing.get("initiation_interval", 1),
                f"{candidate_path}.timing.initiation_interval",
                1,
            )
            input_tail = _require_object(
                timing.get("input_tail", {}),
                f"{candidate_path}.timing.input_tail",
            )
            for edge_id, cycles in input_tail.items():
                _require_string(edge_id, f"{candidate_path}.timing.input_tail key")
                _require_int(
                    cycles, f"{candidate_path}.timing.input_tail.{edge_id}", 0
                )
                if cycles > last_output:
                    raise ProblemError(
                        f"{candidate_path}.timing.input_tail.{edge_id} must not "
                        "exceed last_output"
                    )

            _validate_resources(
                candidate.get("resources", {}),
                f"{candidate_path}.resources",
                resource_limits,
            )
            interfaces = _require_object(
                candidate.get("interfaces"), f"{candidate_path}.interfaces"
            )
            for direction in ("inputs", "outputs"):
                directed = _require_object(
                    interfaces.get(direction, {}),
                    f"{candidate_path}.interfaces.{direction}",
                )
                for edge_id, interface in directed.items():
                    _require_string(edge_id, f"{candidate_path}.{direction} key")
                    _validate_interface(
                        interface, f"{candidate_path}.interfaces.{direction}.{edge_id}"
                    )

    edges = _require_list(problem.get("edges", []), "edges")
    edge_by_id: dict[str, dict[str, Any]] = {}
    for edge_position, raw_edge in enumerate(edges):
        edge_path = f"edges[{edge_position}]"
        edge = _require_object(raw_edge, edge_path)
        edge_id = _require_string(edge.get("id"), f"{edge_path}.id")
        if edge_id in edge_by_id:
            raise ProblemError(f"duplicate edge id {edge_id!r}")
        edge_by_id[edge_id] = edge

        source = _require_int(edge.get("source"), f"{edge_path}.source", 0)
        target = _require_int(edge.get("target"), f"{edge_path}.target", 0)
        if source not in node_by_id or target not in node_by_id:
            raise ProblemError(f"edge {edge_id} references an unknown node")

        modes = _require_list(edge.get("modes"), f"{edge_path}.modes")
        if not modes:
            raise ProblemError(f"edge {edge_id} has no implementation modes")
        seen_mode_ids: set[str] = set()
        for mode_position, raw_mode in enumerate(modes):
            mode_path = f"{edge_path}.modes[{mode_position}]"
            mode = _require_object(raw_mode, mode_path)
            mode_id = _require_string(mode.get("id"), f"{mode_path}.id")
            if mode_id in seen_mode_ids:
                raise ProblemError(f"edge {edge_id} has duplicate mode id {mode_id!r}")
            seen_mode_ids.add(mode_id)
            kind = _require_string(mode.get("kind"), f"{mode_path}.kind")
            if kind not in ("stream", "buffer", "adapter"):
                raise ProblemError(
                    f"{mode_path}.kind must be 'stream', 'buffer', or 'adapter'"
                )
            start_event = mode.get("start_event")
            if start_event not in ("first_output", "last_output"):
                raise ProblemError(
                    f"{mode_path}.start_event must be 'first_output' or 'last_output'"
                )
            if not isinstance(mode.get("couple_completion"), bool):
                raise ProblemError(f"{mode_path}.couple_completion must be boolean")
            if kind == "stream" and (
                start_event != "first_output" or not mode["couple_completion"]
            ):
                raise ProblemError(
                    f"{mode_path} stream modes must start at first_output and "
                    "couple completion"
                )
            if kind == "buffer" and start_event != "last_output":
                raise ProblemError(
                    f"{mode_path} buffer modes must start at last_output"
                )
            _require_int(mode.get("latency", 0), f"{mode_path}.latency", 0)
            _validate_resources(
                mode.get("resources", {}),
                f"{mode_path}.resources",
                resource_limits,
            )
            compatibility = _require_list(
                mode.get("compatibility"), f"{mode_path}.compatibility"
            )
            if not compatibility:
                raise ProblemError(f"{mode_path}.compatibility must not be empty")
            seen_compatibility: set[tuple[str, str]] = set()
            for pair_position, pair in enumerate(compatibility):
                pair_path = f"{mode_path}.compatibility[{pair_position}]"
                pair = _require_list(pair, pair_path)
                if len(pair) != 2:
                    raise ProblemError(
                        f"{pair_path} must be [producer_class, consumer_class]"
                    )
                producer_class = _require_string(pair[0], f"{pair_path}[0]")
                consumer_class = _require_string(pair[1], f"{pair_path}[1]")
                key = (producer_class, consumer_class)
                if key in seen_compatibility:
                    raise ProblemError(f"{pair_path} duplicates compatibility {key}")
                seen_compatibility.add(key)

    order, incoming, outgoing = _topological_order(set(node_by_id), edges)

    for node_id, node in node_by_id.items():
        expected_inputs = set(incoming[node_id])
        expected_outputs = set(outgoing[node_id])
        for candidate in node["candidates"]:
            candidate_id = candidate["id"]
            actual_inputs = set(candidate["interfaces"].get("inputs", {}))
            actual_outputs = set(candidate["interfaces"].get("outputs", {}))
            if actual_inputs != expected_inputs:
                raise ProblemError(
                    f"node {node_id} candidate {candidate_id} input interfaces must "
                    f"match incoming edges {sorted(expected_inputs)}"
                )
            if actual_outputs != expected_outputs:
                raise ProblemError(
                    f"node {node_id} candidate {candidate_id} output interfaces must "
                    f"match outgoing edges {sorted(expected_outputs)}"
                )
            input_tail = candidate["timing"].get("input_tail", {})
            if set(input_tail) != expected_inputs:
                raise ProblemError(
                    f"node {node_id} candidate {candidate_id} input_tail entries must "
                    f"match incoming edges {sorted(expected_inputs)}"
                )

    edge_allowed_tuples: dict[str, list[tuple[int, int, int]]] = {}
    for edge in edges:
        edge_id = edge["id"]
        source = edge["source"]
        target = edge["target"]
        tuples: list[tuple[int, int, int]] = []
        for source_position, source_candidate in enumerate(
            node_by_id[source]["candidates"]
        ):
            producer_class = source_candidate["interfaces"]["outputs"][edge_id][
                "class"
            ]
            for target_position, target_candidate in enumerate(
                node_by_id[target]["candidates"]
            ):
                consumer_class = target_candidate["interfaces"]["inputs"][edge_id][
                    "class"
                ]
                for mode_position, mode in enumerate(edge["modes"]):
                    compatible = {
                        (pair[0], pair[1]) for pair in mode["compatibility"]
                    }
                    if (producer_class, consumer_class) in compatible:
                        tuples.append(
                            (source_position, target_position, mode_position)
                        )
        if not tuples:
            raise ProblemError(
                f"edge {edge_id} has no compatible candidate/mode combination"
            )
        edge_allowed_tuples[edge_id] = tuples

    problem["_objective_mode"] = objective_mode
    problem["_node_by_id"] = node_by_id
    problem["_edge_by_id"] = edge_by_id
    problem["_candidate_index"] = candidate_index
    problem["_edge_allowed_tuples"] = edge_allowed_tuples
    problem["_topological_order"] = order
    problem["_incoming"] = incoming
    problem["_outgoing"] = outgoing
    return problem


def _selected_element(model: Any, choice: Any, values: Iterable[int], name: str) -> Any:
    values = list(values)
    if len(set(values)) == 1:
        return model.new_constant(values[0])
    target = model.new_int_var(min(values), max(values), name)
    model.add_element(choice, values, target)
    return target


def _compute_horizon(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> int:
    horizon = 1
    for node in nodes:
        horizon += max(
            candidate["timing"]["last_output"]
            for candidate in node["candidates"]
        )
    for edge in edges:
        horizon += max(mode.get("latency", 0) for mode in edge["modes"])
    if horizon * LEXICOGRAPHIC_SCALE >= INT64_SAFE_MAX:
        raise ProblemError("timing horizon exceeds the CP-SAT int64 safety limit")
    return horizon


def solve_problem(problem: Any, log_search: bool = False) -> dict[str, Any]:
    """Solve a validated Stream-HLS DSE problem with OR-Tools CP-SAT."""

    problem = validate_problem(problem)
    try:
        import ortools
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - exercised by the CLI path.
        raise RuntimeError(
            "OR-Tools is not installed; run `pip install -r requirements.txt`"
        ) from exc

    nodes = problem["nodes"]
    edges = problem["edges"]
    node_by_id = problem["_node_by_id"]
    incoming = problem["_incoming"]
    outgoing = problem["_outgoing"]
    order = problem["_topological_order"]
    horizon = _compute_horizon(nodes, edges)

    model = cp_model.CpModel()
    choice: dict[int, Any] = {}
    first_offset: dict[int, Any] = {}
    last_offset: dict[int, Any] = {}
    start: dict[int, Any] = {}
    first_output: dict[int, Any] = {}
    last_output: dict[int, Any] = {}
    node_resource: dict[tuple[int, str], Any] = {}

    for node in nodes:
        node_id = node["id"]
        candidates = node["candidates"]
        choice[node_id] = model.new_int_var(
            0, len(candidates) - 1, f"choice_{node_id}"
        )
        first_offset[node_id] = _selected_element(
            model,
            choice[node_id],
            (candidate["timing"]["first_output"] for candidate in candidates),
            f"first_offset_{node_id}",
        )
        last_offset[node_id] = _selected_element(
            model,
            choice[node_id],
            (candidate["timing"]["last_output"] for candidate in candidates),
            f"last_offset_{node_id}",
        )
        for resource in problem["resource_limits"]:
            node_resource[node_id, resource] = _selected_element(
                model,
                choice[node_id],
                (
                    candidate.get("resources", {}).get(resource, 0)
                    for candidate in candidates
                ),
                f"node_{resource}_{node_id}",
            )
        start[node_id] = model.new_int_var(0, horizon, f"start_{node_id}")
        first_output[node_id] = model.new_int_var(
            0, horizon, f"first_output_{node_id}"
        )
        last_output[node_id] = model.new_int_var(
            0, horizon, f"last_output_{node_id}"
        )

    mode_choice: dict[str, Any] = {}
    mode_literals: dict[tuple[str, int], Any] = {}
    edge_trigger: dict[str, Any] = {}
    edge_coupled_end: dict[str, Any] = {}
    edge_resource: dict[tuple[str, str], Any] = {}

    for edge in edges:
        edge_id = edge["id"]
        source = edge["source"]
        target = edge["target"]
        modes = edge["modes"]
        mode_choice[edge_id] = model.new_int_var(
            0, len(modes) - 1, f"mode_{edge_id}"
        )
        literals = []
        for mode_position, mode in enumerate(modes):
            literal = model.new_bool_var(f"mode_{edge_id}_{mode_position}")
            mode_literals[edge_id, mode_position] = literal
            literals.append(literal)
            model.add(mode_choice[edge_id] == mode_position).only_enforce_if(literal)
        model.add_exactly_one(literals)
        model.add_allowed_assignments(
            [choice[source], choice[target], mode_choice[edge_id]],
            problem["_edge_allowed_tuples"][edge_id],
        )

        edge_trigger[edge_id] = model.new_int_var(0, horizon, f"trigger_{edge_id}")
        edge_coupled_end[edge_id] = model.new_int_var(
            0, horizon, f"coupled_end_{edge_id}"
        )
        target_candidates = node_by_id[target]["candidates"]
        input_tail = _selected_element(
            model,
            choice[target],
            (
                candidate["timing"]["input_tail"][edge_id]
                for candidate in target_candidates
            ),
            f"input_tail_{edge_id}",
        )
        for mode_position, mode in enumerate(modes):
            literal = mode_literals[edge_id, mode_position]
            event = (
                first_output[source]
                if mode["start_event"] == "first_output"
                else last_output[source]
            )
            model.add(
                edge_trigger[edge_id] == event + mode.get("latency", 0)
            ).only_enforce_if(literal)
            if mode["couple_completion"]:
                model.add(
                    edge_coupled_end[edge_id]
                    == last_output[source] + mode.get("latency", 0) + input_tail
                ).only_enforce_if(literal)
            else:
                model.add(edge_coupled_end[edge_id] == 0).only_enforce_if(literal)

        for resource in problem["resource_limits"]:
            edge_resource[edge_id, resource] = _selected_element(
                model,
                mode_choice[edge_id],
                (
                    mode.get("resources", {}).get(resource, 0)
                    for mode in modes
                ),
                f"edge_{resource}_{edge_id}",
            )

    for node_id in order:
        if incoming[node_id]:
            model.add_max_equality(
                start[node_id],
                [edge_trigger[edge_id] for edge_id in incoming[node_id]],
            )
        else:
            model.add(start[node_id] == 0)
        model.add(first_output[node_id] == start[node_id] + first_offset[node_id])
        local_completion = model.new_int_var(
            0, horizon, f"local_completion_{node_id}"
        )
        model.add(local_completion == start[node_id] + last_offset[node_id])
        completion_terms = [local_completion]
        completion_terms.extend(
            edge_coupled_end[edge_id] for edge_id in incoming[node_id]
        )
        model.add_max_equality(last_output[node_id], completion_terms)

    resource_usage: dict[str, Any] = {}
    for resource, limit in problem["resource_limits"].items():
        resource_usage[resource] = model.new_int_var(0, limit, f"total_{resource}")
        terms = [node_resource[node["id"], resource] for node in nodes]
        terms.extend(edge_resource[edge["id"], resource] for edge in edges)
        model.add(resource_usage[resource] == sum(terms))

    sinks = [node_id for node_id in order if not outgoing[node_id]]
    latency = model.new_int_var(0, horizon, "latency")
    model.add_max_equality(latency, [last_output[node_id] for node_id in sinks])

    pressure = model.new_int_var(
        0, RESOURCE_PRESSURE_SCALE, "resource_pressure_ppm"
    )
    positive_limits = [
        (resource, limit)
        for resource, limit in problem["resource_limits"].items()
        if limit > 0
    ]
    if positive_limits:
        for resource, limit in positive_limits:
            model.add(
                resource_usage[resource] * RESOURCE_PRESSURE_SCALE
                <= pressure * limit
            )
    else:
        model.add(pressure == 0)

    if problem["_objective_mode"] == "latency":
        model.minimize(latency)
    else:
        model.minimize(latency * LEXICOGRAPHIC_SCALE + pressure)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = problem.get("time_limit_seconds", 60)
    solver.parameters.random_seed = problem.get("random_seed", 0)
    solver.parameters.num_search_workers = problem.get("num_workers", 1)
    solver.parameters.log_search_progress = log_search
    status_code = solver.solve(model)
    status = solver.status_name(status_code).upper()
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "problem_id": problem.get("problem_id"),
        "status": status,
        "solver": {
            "name": "ortools-cp-sat",
            "version": ortools.__version__,
            "wall_time_seconds": solver.wall_time,
            "num_conflicts": solver.num_conflicts,
            "num_branches": solver.num_branches,
            "random_seed": problem.get("random_seed", 0),
            "num_workers": problem.get("num_workers", 1),
        },
        "objective": None,
        "node_selections": [],
        "edge_selections": [],
        "resource_usage": None,
        "timing": None,
    }
    if not has_solution:
        return result

    latency_value = solver.value(latency)
    used_resources = {
        resource: solver.value(total) for resource, total in resource_usage.items()
    }
    pressure_value = max(
        (
            (used_resources[resource] * RESOURCE_PRESSURE_SCALE + limit - 1)
            // limit
            for resource, limit in positive_limits
        ),
        default=0,
    )
    composite_value = (
        latency_value
        if problem["_objective_mode"] == "latency"
        else latency_value * LEXICOGRAPHIC_SCALE + pressure_value
    )
    composite_bound = int(round(solver.best_objective_bound))
    result["objective"] = {
        "mode": problem["_objective_mode"],
        "latency": latency_value,
        "resource_pressure_ppm": pressure_value,
        "composite_value": composite_value,
        "composite_best_bound": composite_bound,
        "relative_gap": (composite_value - composite_bound)
        / max(1, abs(composite_value)),
    }

    for node in nodes:
        node_id = node["id"]
        candidate = node["candidates"][solver.value(choice[node_id])]
        result["node_selections"].append(
            {
                "node_id": node_id,
                "candidate_id": candidate["id"],
                "schedule": candidate.get("schedule", {}),
                "timing": candidate["timing"],
                "resources": candidate.get("resources", {}),
                "interfaces": candidate["interfaces"],
            }
        )
    for edge in edges:
        edge_id = edge["id"]
        mode = edge["modes"][solver.value(mode_choice[edge_id])]
        result["edge_selections"].append(
            {
                "edge_id": edge_id,
                "mode_id": mode["id"],
                "kind": mode["kind"],
                "latency": mode.get("latency", 0),
                "resources": mode.get("resources", {}),
            }
        )
    result["resource_usage"] = {
        resource: {
            "used": used_resources[resource],
            "limit": limit,
            "utilization_ppm": (
                (used_resources[resource] * RESOURCE_PRESSURE_SCALE + limit - 1)
                // limit
                if limit > 0
                else 0
            ),
        }
        for resource, limit in problem["resource_limits"].items()
    }
    result["timing"] = {
        "latency": latency_value,
        "nodes": [
            {
                "node_id": node_id,
                "start": solver.value(start[node_id]),
                "first_output": solver.value(first_output[node_id]),
                "last_output": solver.value(last_output[node_id]),
            }
            for node_id in order
        ],
        "edges": [
            {
                "edge_id": edge["id"],
                "mode_id": edge["modes"][
                    solver.value(mode_choice[edge["id"]])
                ]["id"],
                "trigger": solver.value(edge_trigger[edge["id"]]),
            }
            for edge in edges
        ],
    }
    return result


def _public_problem(problem: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in problem.items() if not key.startswith("_")}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary_path = output.name
    os.replace(temporary_path, path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path, help="DseProblemV1 JSON file")
    parser.add_argument("solution", type=Path, nargs="?", help="DseSolutionV1 JSON file")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the problem without loading OR-Tools",
    )
    parser.add_argument("--log-search", action="store_true", help="enable CP-SAT logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with args.problem.open("r", encoding="utf-8") as source:
            problem = json.load(source)
        validated = validate_problem(problem)
        if args.validate_only:
            print(json.dumps(_public_problem(validated), sort_keys=True))
            return 0
        if args.solution is None:
            raise ProblemError("solution path is required unless --validate-only is used")
        result = solve_problem(problem, log_search=args.log_search)
        _write_json_atomic(args.solution, result)
        if result["status"] in ("OPTIMAL", "FEASIBLE"):
            return 0
        if result["status"] == "INFEASIBLE":
            return 4
        if result["status"] == "MODEL_INVALID":
            return 6
        return 5
    except (OSError, json.JSONDecodeError, ProblemError) as exc:
        print(f"streamhls-dse: invalid problem: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"streamhls-dse: dependency error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
