import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools" / "streamhls-dse"))

import streamhls_dse


try:
    import ortools  # noqa: F401

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


def interface(interface_class="row-major-stream"):
    return {
        "class": interface_class,
        "shape": [16, 16],
        "layout": "row-major",
        "token_order": "i,j",
        "rate": {"tokens": 1, "cycles": 1},
    }


def two_node_problem(dsp_limit=2, bram_limit=1, include_fifo=True):
    modes = []
    if include_fifo:
        modes.append(
            {
                "id": "fifo",
                "kind": "stream",
                "start_event": "first_output",
                "couple_completion": True,
                "latency": 0,
                "resources": {"bram": 1},
                "compatibility": [["row-major-stream", "row-major-stream"]],
            }
        )
    modes.append(
        {
            "id": "buffer",
            "kind": "buffer",
            "start_event": "last_output",
            "couple_completion": False,
            "latency": 0,
            "resources": {},
            "compatibility": [["row-major-stream", "row-major-stream"]],
        }
    )
    return {
        "schema_version": 1,
        "problem_id": "two-node",
        "time_limit_seconds": 10,
        "random_seed": 0,
        "num_workers": 1,
        "objective": {"mode": "lexicographic_latency_resource_pressure"},
        "resource_limits": {"dsp": dsp_limit, "bram": bram_limit},
        "nodes": [
            {
                "id": 0,
                "candidates": [
                    {
                        "id": 10,
                        "schedule": {
                            "permutation": [0],
                            "tiling_factors": [1],
                        },
                        "timing": {
                            "first_output": 2,
                            "last_output": 10,
                            "initiation_interval": 1,
                            "input_tail": {},
                        },
                        "resources": {"dsp": 1},
                        "interfaces": {
                            "inputs": {},
                            "outputs": {"edge-0-1": interface()},
                        },
                    },
                    {
                        "id": 11,
                        "schedule": {
                            "permutation": [0],
                            "tiling_factors": [4],
                        },
                        "timing": {
                            "first_output": 1,
                            "last_output": 4,
                            "initiation_interval": 1,
                            "input_tail": {},
                        },
                        "resources": {"dsp": 4},
                        "interfaces": {
                            "inputs": {},
                            "outputs": {"edge-0-1": interface()},
                        },
                    },
                ],
            },
            {
                "id": 1,
                "candidates": [
                    {
                        "id": 20,
                        "schedule": {
                            "permutation": [0],
                            "tiling_factors": [1],
                        },
                        "timing": {
                            "first_output": 0,
                            "last_output": 5,
                            "initiation_interval": 1,
                            "input_tail": {"edge-0-1": 0},
                        },
                        "resources": {"dsp": 1},
                        "interfaces": {
                            "inputs": {"edge-0-1": interface()},
                            "outputs": {},
                        },
                    }
                ],
            },
        ],
        "edges": [
            {
                "id": "edge-0-1",
                "source": 0,
                "target": 1,
                "modes": modes,
            }
        ],
    }


def tied_single_node_problem():
    problem = two_node_problem(dsp_limit=4)
    problem["nodes"] = [
        {
            "id": 0,
            "candidates": [
                {
                    "id": 1,
                    "timing": {
                        "first_output": 2,
                        "last_output": 10,
                        "input_tail": {},
                    },
                    "resources": {"dsp": 4},
                    "interfaces": {"inputs": {}, "outputs": {}},
                },
                {
                    "id": 2,
                    "timing": {
                        "first_output": 2,
                        "last_output": 10,
                        "input_tail": {},
                    },
                    "resources": {"dsp": 2},
                    "interfaces": {"inputs": {}, "outputs": {}},
                },
            ],
        }
    ]
    problem["edges"] = []
    problem["resource_limits"] = {"dsp": 4}
    return problem


class ValidationTests(unittest.TestCase):
    def test_accepts_valid_problem_without_solver(self):
        validated = streamhls_dse.validate_problem(two_node_problem())
        self.assertEqual(validated["_topological_order"], [0, 1])
        self.assertEqual(len(validated["_edge_allowed_tuples"]["edge-0-1"]), 4)

    def test_rejects_missing_interface_contract(self):
        problem = two_node_problem()
        del problem["nodes"][1]["candidates"][0]["interfaces"]["inputs"][
            "edge-0-1"
        ]
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "input interfaces"):
            streamhls_dse.validate_problem(problem)

    def test_rejects_missing_input_tail(self):
        problem = two_node_problem()
        problem["nodes"][1]["candidates"][0]["timing"]["input_tail"] = {}
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "input_tail"):
            streamhls_dse.validate_problem(problem)

    def test_rejects_interface_without_compatible_mode(self):
        problem = two_node_problem()
        problem["nodes"][1]["candidates"][0]["interfaces"]["inputs"][
            "edge-0-1"
        ]["class"] = "column-major-stream"
        with self.assertRaisesRegex(
            streamhls_dse.ProblemError, "no compatible candidate/mode"
        ):
            streamhls_dse.validate_problem(problem)

    def test_rejects_undeclared_resource(self):
        problem = two_node_problem()
        problem["edges"][0]["modes"][0]["resources"]["uram"] = 1
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "without a declared"):
            streamhls_dse.validate_problem(problem)

    def test_rejects_input_tail_larger_than_candidate_latency(self):
        problem = two_node_problem()
        problem["nodes"][1]["candidates"][0]["timing"]["input_tail"][
            "edge-0-1"
        ] = 6
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "exceed last_output"):
            streamhls_dse.validate_problem(problem)

    def test_rejects_unsafe_stream_timing_contract(self):
        problem = two_node_problem()
        problem["edges"][0]["modes"][0]["couple_completion"] = False
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "stream modes"):
            streamhls_dse.validate_problem(problem)

    def test_rejects_cycles(self):
        problem = two_node_problem()
        problem["edges"].append(
            {
                "id": "edge-1-0",
                "source": 1,
                "target": 0,
                "modes": [
                    {
                        "id": "fifo",
                        "kind": "stream",
                        "start_event": "first_output",
                        "couple_completion": True,
                        "latency": 0,
                        "resources": {},
                        "compatibility": [
                            ["row-major-stream", "row-major-stream"]
                        ],
                    }
                ],
            }
        )
        with self.assertRaisesRegex(streamhls_dse.ProblemError, "acyclic"):
            streamhls_dse.validate_problem(problem)


@unittest.skipUnless(HAS_ORTOOLS, "OR-Tools is not installed")
class SolverTests(unittest.TestCase):
    def test_resource_budget_selects_low_dsp_candidate(self):
        result = streamhls_dse.solve_problem(two_node_problem(dsp_limit=2))
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective"]["latency"], 10)
        self.assertEqual(result["node_selections"][0]["candidate_id"], 10)
        self.assertEqual(result["edge_selections"][0]["mode_id"], "fifo")

    def test_larger_budget_selects_faster_candidate(self):
        result = streamhls_dse.solve_problem(two_node_problem(dsp_limit=5))
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective"]["latency"], 6)
        self.assertEqual(result["node_selections"][0]["candidate_id"], 11)

    def test_buffer_waits_for_source_completion(self):
        result = streamhls_dse.solve_problem(
            two_node_problem(dsp_limit=2, include_fifo=False)
        )
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective"]["latency"], 15)
        self.assertEqual(result["edge_selections"][0]["mode_id"], "buffer")

    def test_edge_resources_participate_in_global_budget(self):
        result = streamhls_dse.solve_problem(
            two_node_problem(dsp_limit=2, bram_limit=0)
        )
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["edge_selections"][0]["mode_id"], "buffer")
        self.assertEqual(result["resource_usage"]["bram"]["used"], 0)

    def test_lexicographic_tie_selects_lower_resource_pressure(self):
        result = streamhls_dse.solve_problem(tied_single_node_problem())
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective"]["latency"], 10)
        self.assertEqual(result["objective"]["resource_pressure_ppm"], 500_000)
        self.assertEqual(result["node_selections"][0]["candidate_id"], 2)

    def test_infeasible_budget_has_no_selections(self):
        result = streamhls_dse.solve_problem(two_node_problem(dsp_limit=1))
        self.assertEqual(result["status"], "INFEASIBLE")
        self.assertEqual(result["node_selections"], [])
        self.assertEqual(result["edge_selections"], [])

    def test_cli_writes_structured_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            problem_path = Path(directory) / "problem.json"
            solution_path = Path(directory) / "solution.json"
            problem_path.write_text(
                json.dumps(two_node_problem()), encoding="utf-8"
            )
            exit_code = streamhls_dse.main(
                [str(problem_path), str(solution_path)]
            )
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(solution["status"], "OPTIMAL")
        self.assertEqual(solution["objective"]["latency"], 10)


if __name__ == "__main__":
    unittest.main()
