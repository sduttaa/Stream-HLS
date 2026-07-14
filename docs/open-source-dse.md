# Open-source DSE foundation

Stream-HLS is migrating from the AMPL/Gurobi optimizer embedded in `DFG.cpp`
to a solver-neutral, versioned design-space optimization (DSE) interface. The
default implementation uses OR-Tools CP-SAT and requires no license server.

This first implementation is deliberately a restricted, exact master problem.
It composes a finite set of verified node candidates and communication modes.
Later optimizers can generate candidate columns without changing the compiler
or solution contract.

## Process boundary

The compiler will emit `DseProblemV1` JSON, execute `streamhls-dse`, validate
`DseSolutionV1`, and only then mutate MLIR. Solver stdout is diagnostic output;
it is never parsed as the result channel.

The boundary has four important properties:

- candidate IDs are stable external identifiers;
- resources are named vectors rather than DSP-only values;
- communication is an explicit implementation with its own timing/resources;
- interfaces are contracts, so compatibility is independent of candidate IDs.

The executable currently establishes and tests this boundary. Emission and
solution ingestion from `DFG.cpp` are the next integration step; the existing
AMPL path remains unchanged until end-to-end parity is demonstrated.

## Objective

The default mode is `lexicographic_latency_resource_pressure`:

1. minimize graph latency;
2. among equal-latency designs, minimize the maximum normalized utilization
   over all resources with a non-zero limit.

Resource pressure is represented in parts per million:

```text
pressure = max_resource ceil(used[resource] * 1,000,000 / limit[resource])
```

The CP-SAT objective is exactly lexicographic because latency is multiplied by
`1,000,001`. One cycle is therefore more important than the entire secondary
range. Set `objective.mode` to `latency` when resource tie-breaking is not
desired.

This is not a weighted average of DSP, LUT, BRAM, and other resources. Every
resource remains a hard independent constraint.

## Node candidates

Each node contains a local Pareto set of implementations. A candidate records:

- a stable ID and optional replayable schedule metadata;
- relative first-output and last-output cycles;
- initiation interval and per-input tail cycles;
- a named resource vector;
- input and output interface contracts.

An interface has a required `class` and may additionally record shape, layout,
token order, and token rate. The class is the compatibility key used by the
master; the remaining fields provide a checkable contract for the compiler.

```json
{
  "id": 10,
  "schedule": {
    "permutation": [0, 1, 2],
    "tiling_factors": [1, 1, 8]
  },
  "timing": {
    "first_output": 8,
    "last_output": 1024,
    "initiation_interval": 1,
    "input_tail": {}
  },
  "resources": {"dsp": 24, "bram": 2},
  "interfaces": {
    "inputs": {},
    "outputs": {
      "edge-0-1": {
        "class": "row-major-f32-rate1",
        "shape": [32, 32],
        "layout": "row-major",
        "token_order": "i,j",
        "rate": {"tokens": 1, "cycles": 1}
      }
    }
  }
}
```

Every candidate must have exactly one interface entry for each incident edge.
For every incoming edge, `timing.input_tail[edge]` is the number of cycles from
the final input token becoming available to the candidate's final output.

## Communication modes

An edge supplies alternative implementations such as FIFO streaming, shared
buffering, or an adapter/reorder unit:

```json
{
  "id": "edge-0-1",
  "source": 0,
  "target": 1,
  "modes": [
    {
      "id": "fifo",
      "kind": "stream",
      "start_event": "first_output",
      "couple_completion": true,
      "latency": 0,
      "resources": {"bram": 1},
      "compatibility": [
        ["row-major-f32-rate1", "row-major-f32-rate1"]
      ]
    },
    {
      "id": "shared-buffer",
      "kind": "buffer",
      "start_event": "last_output",
      "couple_completion": false,
      "latency": 1,
      "resources": {"bram": 4},
      "compatibility": [
        ["row-major-f32-rate1", "row-major-f32-rate1"]
      ]
    }
  ]
}
```

`start_event` determines whether the consumer can start at the producer's
first or final output. When `couple_completion` is true, the consumer cannot
finish before the producer's last output plus link latency and the selected
consumer's input-tail cycles. This captures pipeline fill/drain behavior more
faithfully than a single streaming boolean.

Communication resources participate in the same global budgets as compute
resources. A FIFO that consumes BRAM can therefore make an otherwise attractive
streaming design infeasible.

## Complete problem outline

```json
{
  "schema_version": 1,
  "problem_id": "gemm-pipeline",
  "time_limit_seconds": 60,
  "random_seed": 0,
  "num_workers": 1,
  "objective": {
    "mode": "lexicographic_latency_resource_pressure"
  },
  "resource_limits": {
    "dsp": 2560,
    "lut": 650000,
    "ff": 1300000,
    "bram": 1500,
    "uram": 600
  },
  "nodes": [],
  "edges": []
}
```

The implementation performs semantic validation in addition to ordinary JSON
shape checks. It rejects cycles, unknown resources, incomplete interface/tail
contracts, duplicate IDs, and edges with no compatible candidate/mode tuple.

## Timing model

For a selected node candidate:

```text
first_output(node) = start(node) + candidate.first_output
local_end(node)    = start(node) + candidate.last_output
```

Each edge trigger is its selected producer event plus link latency. A node
starts at the latest incoming trigger. Its final output is the maximum of its
local end and every completion-coupled producer end plus input-tail cycles.
Graph latency is the latest sink final-output cycle.

This model handles first-token overlap and last-token coupling, but it is not
yet a general timed-SDF model. Explicit rates are recorded now so a future
version can add system II, batch fill/drain timing, rate adapters, and derived
FIFO depths without changing interface identity.

## Solution and status contract

`OPTIMAL` and `FEASIBLE` contain a usable incumbent. `INFEASIBLE`, `UNKNOWN`,
and `MODEL_INVALID` contain no node or edge selections and must stop
compilation. A solution records:

- latency and resource pressure;
- the composite best bound and relative gap;
- selected node candidates and edge modes;
- total resource use and per-resource utilization;
- node/edge timing;
- solver name/version, seed, worker count, conflicts, branches, and wall time.

The command returns:

- `0` for `OPTIMAL`, `FEASIBLE`, or successful validation;
- `2` for malformed input;
- `3` for a missing solver dependency;
- `4` for `INFEASIBLE`;
- `5` for `UNKNOWN` or another no-incumbent status;
- `6` for `MODEL_INVALID`.

## Developer commands

```bash
python3 tools/streamhls-dse/streamhls-dse problem.json solution.json
python3 tools/streamhls-dse/streamhls-dse --validate-only problem.json
python3 -m unittest discover -s test/python -p 'test_*.py' -v
```

`num_workers` defaults to one so CI results are reproducible. Production runs
may explicitly request more workers and must record that setting in the compile
manifest.

## Deliberate non-goals of this foundation

- It does not yet generate candidate frontiers from MLIR.
- It does not replace the existing AMPL call until compiler integration and
  benchmark parity are complete.
- It does not claim physical optimality from analytical timing.
- It does not yet implement column generation, FIFO-depth proofs, learned
  residuals, or synthesis/place-and-route feedback.

Those capabilities build on this schema; they do not belong in the first
license-removal change.
