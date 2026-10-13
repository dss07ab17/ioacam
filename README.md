# iOACAM Workflow Engine

Judges observation events against a declared workflow policy. No cameras, no
models, no hardware. Pure logic, stdlib only, Python 3.10+.

```bash
python3 harness/runner.py          # 19 scenarios
python3 harness/runner.py -v       # with every finding printed
python3 harness/test_engine.py     # property tests
python3 harness/test_lint.py       # linter tests
python3 identity/test_identity.py  # identity cascade tests
python3 identity/test_alignment.py # face alignment tests
python3 identity/test_directory.py # employee directory lookup tests
python3 tools/lint_policy.py workflows/*.json --strict
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/test_yolo26.py
python3 perception/test_ppe.py      # PPE detector, thresholds, attribution
python3 perception/test_stages.py   # opt-in stage wiring
python3 perception/test_intervals.py # action interval join
```

All suites pass; the example policy lints clean.

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
  scenarios/        19 JSON scenarios
tools/
  lint_policy.py       catches what schema validation cannot
  inspect_checkpoint.py  read a .pth's input contract without torch
perception/
  perceive.py       production events pipeline (see perception/README.md)
  objects.py        object tracks + object_at_station (shared with demo)
  ppe_stage.py      second detector for hard hats / vests; ppe_helmet, ppe_vest
  pose.py           RTMPose / stub pose stage (shared)
  actions_stage.py  PoseC3D cadence + action_recognised (shared)
  action_intervals.py joins consecutive windows into action_interval
  identity_stage.py binds tracks to enrolled faces in the loop; opt-in
  association.py    object-to-person hand-anchor attribution
  detectors/        pluggable model backend, chosen by config; default YOLOX,
                    Apache-2.0. YOLO26 (AGPL, NMS-free) alongside for evaluation,
                    and yolo26-ppe, the same decode over a 10-class PPE fine-tune
  face/             RFB-320 face detector behind the same kind of seam
  actions/          RTMPose + PoseC3D plumbing; NTU-60 head is not site-useful yet
identity/
  roster.py         who is where, from access control transactions
  matcher.py        face matcher seam + threshold policy
  resolver.py       the tiered cascade
  arcface_matcher.py  ArcFace embeddings; alignment.py + landmarks.py feed it
  enrollment.py     enrolled templates, keyed on person ID; enroll.py is
                    the CLI over it
  directory.py      employee ID -> name, for the screen only. Reloadable,
                    optional, and never read by anything that writes an event
schema/         the perception <-> engine contract
workflows/      declared policies (manufacturing example + pick-and-place)
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
| 13 two concurrent operators | Each actor gets an instance; neither disturbs the other |
| 14 concurrent one deviates | Only the deviating actor is reported |
| 15 correlation lost on departure | Subject leaves; unresolved steps are unknown, not violations |
| 16 correlation lost on coverage loss | Blindness is never reported as skipped steps |
| 17 instance capacity | Cap reached routes to maintenance, not to the SOC |
| 18 unaccounted presence | Nobody the site can account for is a violation |
| 19 unbadged zone | Known person, zone they never authenticated into |

## Concurrency

A workflow declares `correlation`: an ordered list of event attributes that
decide which instance an event belongs to. Empty means singleton, which is the
original behaviour and still correct for workflows that cannot overlap.

**Order expresses precedence, and it matters.** `subject.identity` before
`track_id`, because a tracker id swaps when two people cross and is reissued
when someone leaves and returns; an enrolled identity survives both.

**Zone fallback exists because some sources have no per-actor identity.** A PLC
reporting a torque cycle knows the station, not the operator. Refusing to
correlate those would leave every bus-evidenced step permanently unprovable, so
events lacking a correlation attribute fall back to zone matching. Where two
instances share a zone this is a guess: the engine assigns to the
longest-waiting instance and reports the ambiguity once per workflow and zone,
rather than on every message. The fix is for bus events to carry a correlation
key, or for the workflow to be keyed on zone.

**Losing the subject is not a violation.** This is the important one. If a
tracker drops someone mid-workflow and the engine failed their remaining steps
normally, a tracking limitation would be reported as a critical violation
against a named person -- a false accusation with a technical cause, and the
fastest way to lose a customer's trust. So correlation loss closes the instance
with a single `unknown` finding routed to review, and the unresolved steps are
named rather than judged.

Three things trigger it: a `person_left_zone` event matching the instance's
key, the zone losing all sensor coverage, or optionally `correlation_timeout_s`
of silence. That last one is off by default, because it is only safe when
perception emits periodic liveness for active tracks; without that, a
legitimately long step is indistinguishable from a lost subject.

**Hitting `max_concurrent_instances` is diagnostic.** Instances are created
from observed events, so a churning tracker would spawn them without bound. The
cap makes that visible instead of silent, and the finding routes to maintenance
because it is a perception problem, not a workflow deviation.

## Identity

Camera tracks are bound to enrolled identities using what the doors already
know, rather than by re-identifying faces at monitoring range. See
`identity/README.md`. Three things from it are worth knowing here:

**Thresholds are derived from candidate count, not chosen.** Comparing against
N candidates multiplies the false match rate by roughly N, so the policy
declares an acceptable rate per decision and derives the per-comparison
threshold as `target / N`.

**That produces a hard ceiling.** Beyond `target / strongest_measured_FMR`
candidates, no operating point is strict enough and whole-site matching stops
working entirely. `ThresholdPolicy.max_candidates()` computes it, and it must
be known at commissioning. This is why the presence TTL and badge-out matter:
they keep the roster small enough to remain decidable.

**Authentication and attribution are different facts.** The door verified a
person who presented themselves; the camera believes a track is that person.
The second carries a match confidence so downstream rules can refuse to act on
a weak one, and the log has to show which link was inferred. If a badge is
lent, everything downstream is wrong.

**The enrolment key is a person ID, never a name.** `EMP-4471`, not `Rajesh`.
Names are not unique and they change; the ID is already what the badge reader
emits, so the roster, the door and the workflow engine all say the same thing
about the same person with no translation step in between. A display name may
be enrolled alongside it for showing on screen, and takes no part in matching.
Events and logs carry the ID only, which is also most of why the event store
holds no personal names on its face.

**Every camera event about a bound track carries the ID.** Not just the three
identity observations: zone entries and exits, `action_recognised`,
`action_interval` and the PPE pair all carry `subject.identity` once
`perception/identity_stage.py` has bound the track, which is what makes
`subject.identity` usable as the correlation key named above. An **unbound**
track carries no `identity` key at all — not an empty one — because
`wrong_role` tests for a role they can act on and would read a placeholder as
an answer. And `track_id` stays beside the identity rather than being replaced
by it: the track id is what the camera observed, the identity is an inference
about who that was, and an attribution has to remain questionable.

**The ID-to-name map is one file, and it is optional.**
`identity/data/employees.csv` -- five columns, gitignored, loaded once and
reloadable -- is what puts a name on screen. Nothing that writes an event reads
it, so the event store and the personal data stay separable and can be
access-controlled and deleted on different schedules. Delete the file and the
system runs on IDs. If something breaks without it, a name has leaked into a
path that should only have carried an ID.

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
property test across every file in `harness/scenarios/`.

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

**Log integrity, the signed half.** `emit.py` implements the monotonic
counter and the hash chain, emitted as `integrity.seq` and `integrity.prev_hash`
when `integrity.enabled` is set. The TPM-keyed signature the schema also
specifies is not implemented, and without it the chain proves the log is
internally consistent, not that this device wrote it.

**Policy signing.** An attacker who edits `workflows/*.json` turns `violation`
into `conformant`. Sign the file, anchor its version in the TPM, stamp
`policy_version` on every finding -- the last of those the engine already does.

**Calibration.** Temperature scaling lives in `perception/confidence.py`.
The engine compares the resulting confidence against policy thresholds, so
`calibration.temperature` in the camera config must match the policy file.

## Next

**Transport.** Findings are produced and go nowhere. MQTT out, per the software
architecture. Until this exists the engine is a library, not a running system.

**Bus inputs.** Robot and machine telemetry over OPC-UA, Modbus or CAN. The
easiest work on the list and the highest value per hour: those events arrive at
confidence 1.0 and they carry the cross-attestation capability nothing in the
market study offers. The robot workflow in the example policy currently has no
real source feeding it.

**Log integrity.** TPM-keyed signing over the chain `emit.py` already
produces. The counter and the hash chain exist; the key that would make them
evidence does not.

**The board.** YOLOX and RTMPose already export to ONNX in this repo
(`perception/tools/fetch_model.py`, `perception/tools/export_rtmpose.py`), and
so does the alternative YOLO26 detector (`perception/tools/export_yolo26.py`),
which is NMS-free and so carries no NMS operator for the converter to reject —
but is AGPL-3.0 and cannot ship.
Confirm conversion through RKNN-Toolkit2 on real hardware, then measure what
NPU headroom remains once face recognition runs alongside. That answer
decides one board or two, and it belongs in the proposal. A model that will
not convert costs nothing to discover in week one and everything in month
four.

## Known issue

`perception/perceive.py` writes events to stdout. Redirecting that with `>` in
PowerShell produces UTF-16, which no JSON Lines reader will accept. Either pipe
through `Set-Content -Encoding utf8`, or add an `--output` flag so the process
opens the file itself with `encoding="utf-8"` and the shell is never involved.
The second is better, and matters more on the board rather than less.
