# iOACAM Workflow Engine

Judges observation events against a declared workflow policy. No cameras, no
models, no hardware. Pure logic, stdlib only, Python 3.10+.

## Quick Start

**First time setup?** See [INSTALLATION.md](INSTALLATION.md) for complete setup instructions including:
- Python environment setup
- Perception layer configuration (YOLOX model download, zone definition)
- Running tests and the full workflow

**Already set up?** Run the test suite:

```bash
python3 harness/runner.py          # 12 scenarios
python3 harness/runner.py -v       # with every finding printed
python3 harness/test_engine.py     # property tests
python3 harness/test_lint.py       # linter tests
python3 tools/lint_policy.py workflows/*.json --strict
```

All suites pass; the example policy lints clean.

**Extending the system?** See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- How to add new observation types
- Building custom detectors
- Creating test scenarios
- Policy development workflow

## Layout

```
engine/
  model.py      enums, Event, Finding
  policy.py     policy loading, validation, response matrix resolution
  instance.py   step lifecycle state machine
  engine.py     ingest / tick / flush
harness/
  runner.py         scenario runner
  test_engine.py    property tests
  test_lint.py      linter tests
  scenarios/        12 JSON scenarios
tools/
  lint_policy.py    catches what schema validation cannot
schema/         the perception <-> engine contract
workflows/      the declared policy (example: plant A, line 3)
perception/     separate process: webcam -> YOLOX -> zones -> events
```

## Event Format

Perception emits JSON lines (newline-delimited JSON) to stdout, one event per line:

```json
{"event_id":"2c61435b-32cc-40ce-8abb-56f76ec4b13b","timestamp_us":976786338561,"wall_time":"2026-08-31T18:08:52.114Z","source":"camera","sensor_id":"cam-01","observation":"person_in_zone","confidence":0.7257,"track_id":"trk-0001","zone_id":"zone-assembly-4","value":true,"subject":{"class":"human"},"confidence_components":{"raw_score":0.9067,"calibrated_score":0.7796,"quality":0.9309,"persistence":1.0,"agreement":1.0,"frames_observed":4}}
```

Each event contains:
- **Observation type** — `person_in_zone`, `person_left_zone`, `person_count`, `sensor_health`
- **Confidence** — 0.0–1.0, composed of raw score, calibration, quality, persistence, agreement
- **Identifiers** — `event_id`, `track_id`, `zone_id`, `sensor_id`
- **Timestamps** — microseconds since epoch + ISO 8601 wall clock

All events conform to [schema/event.schema.json](schema/event.schema.json).

To save events to a file:
```bash
python perception/perceive.py --config perception/config/webcam.json > events.jsonl
```

## The three quantities

Computed separately, combined only at the end. Conflating them is the design
error the whole structure exists to prevent.

| | Source | Example |
|---|---|---|
| **Verdict** | Rules, at runtime | `violation` |
| **Severity** | Authored in the policy file | `critical` |
| **Confidence** | Perception only, at runtime | `0.60` |

Scenarios 06 and 10 are the same violation at different confidence. One
alarms and requests mitigation; the other goes to a human with the clip. That
difference is the point.

Events from a bus, badge or timer carry `confidence: 1.0`. There is no
perception uncertainty in a fact, and the engine treats them accordingly.

## What the scenarios cover

| Scenario | Establishes |
|---|---|
| 01 clean run | Conformant operation produces `response: none` |
| 02 skipped step | Omission detected by deadline, not by a detector |
| 03 out of order | Step completed before its predecessor |
| 04 overrun tolerated | Grace band yields `tolerated`, not `violation` |
| 05 underrun | Implausibly fast completion is a violation |
| 06 wrong role | Right action, wrong person, seen clearly |
| 07 wrong zone | Zone rule certain, identification uncertain |
| 08 repeated step | Second completion flagged, not silently accepted |
| 09 unknown | Undeclared observation never becomes conformant |
| 10 low confidence | Same violation as 06, routed to a human |
| 11 robot attestation | Controller claims running, motion never corroborates |
| 12 coverage lost | Blind zone reports unknown, never conformant |

## Design decisions worth preserving

**A failed step still releases its successors.** Otherwise one skip stalls the
instance and every later step reports as skipped too, burying the real finding.

**Open steps get first claim on evidence.** Real policies reuse observations
across steps -- `person_in_zone` proves both step 1 and step 4. Without
precedence, one normal event would complete a future step out of order *and*
flag an earlier one as repeated. See `_apply_to_instance`.

**Successors are promoted at the settle time, not the tick time.** On the
device `tick()` runs every few hundred milliseconds and the difference is
invisible. In a test that jumps sixty seconds it decides whether the next
step's clock is right. `test_engine.py` locks this with a tick-invariance
property test across all twelve scenarios.

**A step's confidence is the weakest link across its evidence.** Taking the
mean would let one certain bus reading paper over a marginal visual detection.

**Wrong role and wrong zone are checked before the confidence gate.** A
low-confidence wrong-role event must surface as a wrong-role finding, not
disappear as an unrelated observation.

**Malformed policies fail at load.** A dangling predecessor or a cycle leaves
steps permanently pending, so the workflow silently stops detecting anything.
That is the worst failure this product has, so it is a load-time error.

**An unmatched finding still resolves.** If the response matrix has a gap, the
finding goes to the review queue rather than vanishing.

**Mitigation is requested, never actuated.** iOACAM is supervisory. Safety
functions stay with light curtains and safety PLCs.

## The linter, and why it exists

A policy can validate perfectly against the schema and still detect nothing.
That failure is silent: the system reports conformant operation forever and
nobody notices until an incident is missed. `tools/lint_policy.py` catches
what schema validation cannot, and `harness/test_lint.py` verifies each check
still fires.

Its own development is the argument for it. Three faults were found in this
repo's example policy, in order:

**Evidence too weak to identify the step.** Visual inspection was proved by
`person_in_zone` alone -- the same observation that proves step 1, so presence
stood in for an inspection happening. Found by tracing the scenarios by hand.
Now caught as `ambiguous-evidence`.

**Tolerance wider than the minimum.** Step 3 had `min_duration_s: 8` and
`duration_tolerance_s: 10`, so every underrun fell inside the grace band and
the critical underrun rule could never fire. Found by hand, fixed on step 3
only.

**The same fault on two more steps.** The linter's first run found steps 2 and
4 broken identically. Fixing one instance by hand and believing the class was
handled is exactly the failure mode a linter exists to prevent.

That third finding then exposed a schema flaw rather than a values flaw: one
`duration_tolerance_s` was serving both bounds. A 15-second grace on a
45-second maximum is sensible; the same 15 seconds against a 3-second minimum
swallows the rule whole. Split into `underrun_tolerance_s` and
`overrun_tolerance_s`, with the shared field kept as a fallback and flagged
as `shared-tolerance` when both bounds exist.

Checks include: unreachable underrun rules, inverted duration bands, required
steps with no deadline (which makes omission undetectable), evidence that
cannot distinguish two steps, observations outside the schema enum, confidence
thresholds that can never be met, typos in `deviation_severity` keys, severities
declared for deviations that cannot occur, workflows with no entry step, zones
with no covering sensor, safety-critical zones with no matching response row,
response rows shadowed by earlier broader rows, and confidence bands with no
row at all.

One deliberate exemption: a vision step whose predecessor supplies telemetry is
the cross-attestation pattern, not a weakly-evidenced critical step, so it is
not flagged.

## Not built yet, deliberately

**Concurrent instances.** One instance per workflow at a time. Multiple
concurrent runs need a correlation key (`track_id`, `asset_id`) and a policy
on what happens when two actors interleave. Out of scope until the single
case is proven at a site.

**Log integrity.** The `integrity` block in `event.schema.json` is specified
but not implemented. Hash chain, monotonic counter, TPM-keyed signature.

**Policy signing.** An attacker who edits `workflows/*.json` turns `violation`
into `conformant`. Sign the file, anchor its version in the TPM, stamp
`policy_version` on every finding -- the last of those the engine already does.

**Calibration.** `calibration.temperature` is loaded and carried but not
applied by the engine; confidence arrives pre-calibrated. That decision is now
made: the correction lives in `perception/confidence.py`. The two values must
be kept equal, or the engine compares confidences against thresholds tuned on a
different scale. Nothing enforces that yet.

## Next

The target perception pipeline:

```
GStreamer -> YOLOX -> ByteTrack -> RTMPose -> ST-GCN -> events -> this engine
```

`perception/` is the first stage of that, running now: OpenCV capture, YOLOX person
detection, a greedy IoU tracker, zone polygons, and calibrated events on
stdout. It is a separate process and imports nothing from `engine/`, so the
board port swaps it and leaves the engine untouched. Four of the nineteen
observations are covered; PPE, pose and action need RTMPose and ST-GCN.

**The detector's licence is a product decision, not a technical one.**
Ultralytics YOLO is AGPL-3.0, which would oblige releasing the engine, the
policies and the site integrations as AGPL. The default is YOLOX (Apache-2.0),
and the backend is a config string so the choice stays reversible. Reasoning,
alternatives and the traps next door: `perception/LICENCE-NOTES.md`.

Before adopting any model, confirm it exports to ONNX and converts through
RKNN-Toolkit2 on the actual board. A model that will not convert costs nothing
to discover in week one and everything in month four. Still unverified for
YOLOX here.

Building the perception layer also found a defect in the contract itself:
`value` used `oneOf` over `boolean / integer / number / string`, and since every
JSON integer is also a number, every integer matched two branches and failed
validation. `person_count` could never validate. The scenarios only ever used
booleans and strings, so nothing had hit it. Now `anyOf`.

## Documentation

| Document | Purpose |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | Environment setup, dependency installation, perception layer configuration |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Extending the system, adding detectors, test scenarios, debugging |
| [schema/](schema/) | Event and workflow schema documentation |
| [perception/README.md](perception/README.md) | Perception layer architecture and zone configuration |
| [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md) | Model licensing and alternatives (YOLOX, YOLO, etc.) |

## Repository Structure

```
.
├── README.md                      # This file
├── INSTALLATION.md                # Setup and configuration
├── DEVELOPMENT.md                 # Development guide
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
│
├── engine/                        # Workflow validation engine
│   ├── engine.py                 # Main engine: ingest, tick, flush
│   ├── instance.py               # Step lifecycle state machine
│   ├── model.py                  # Data models: Event, Finding
│   └── policy.py                 # Policy loading and validation
│
├── perception/                    # Camera detection layer
│   ├── perceive.py               # Main entry point
│   ├── tracking.py               # Person tracking
│   ├── zones.py                  # Zone membership
│   ├── confidence.py             # Confidence calibration
│   ├── detectors/                # Pluggable detector backends
│   ├── models/                   # Pre-trained model weights
│   ├── config/                   # Configuration files
│   └── tools/                    # Utilities (model download, zone definition)
│
├── harness/                       # Test suite
│   ├── runner.py                 # Scenario runner
│   ├── test_engine.py            # Property-based tests
│   ├── test_lint.py              # Linter tests
│   └── scenarios/                # 12 test scenarios (JSON)
│
├── schema/                        # JSON contracts
│   ├── event.schema.json         # Perception event format
│   └── workflow.schema.json      # Policy declaration format
│
├── tools/                         # Utilities
│   ├── lint_policy.py            # Policy validator
│   ├── fetch_model.py            # Download YOLOX model
│   ├── define_zone.py            # Interactive zone tool
│   └── validate_events.py        # Event validation
│
└── workflows/                     # Policy files
    └── example_manufacturing_policy.json
```

## License

YOLOX weights and code are Apache-2.0 (see [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md) for rationale and alternatives).

