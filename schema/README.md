# iOACAM Workflow Monitoring — Schema

The contract between the perception layer and the workflow engine. Settle this first;
both sides can then be built independently.

## Files

| File | Purpose |
|---|---|
| `event.schema.json` | The single message type perception emits and the engine consumes |
| `workflow.schema.json` | Declared workflows, zones, and the response matrix |
| `example_manufacturing_policy.json` | Worked example: assembly station + robot cross-check |

## The boundary

```
  cameras, radar, buses            declared policy
           |                              |
           v                              v
   +---------------+              +----------------+
   |  perception   | -- events -> | workflow engine|
   +---------------+              +----------------+
                                          |
                                          v
                              verdict / severity / confidence
                                          |
                                          v
                                    response matrix
```

Perception never judges. The engine never sees pixels. Two processes, one message
format. On the PC both run locally; on the board only the perception side is
swapped for the RKNN build.

## Three independent quantities

These are computed separately and only combined at the end. Conflating them is the
most common design error.

**Verdict** — from the rules, comparing observation against the declaration.
`conformant` / `tolerated` / `violation` / `unknown`.

**Severity** — authored in the policy file, fixed per step per deviation type. Not
computed at runtime. The same physical event is informational at one step and
critical at another.

**Confidence** — computed at runtime from perception only:

```
confidence = calibrated_score x quality x persistence x agreement
```

Nothing about the verdict or severity affects it.

Events from a bus read, an access-control transaction, or a timer carry
`confidence: 1.0`. There is no perception uncertainty in a fact.

## Design decisions worth keeping

**The observation list is closed.** Adding an entry means committing to a detector,
a feasibility check, and an ATP test. Anything outside the list is what `unknown`
exists for.

**`unknown` is never treated as `conformant`.** A system that silently accepts what it
does not recognise is blind to exactly the behaviour it was bought to catch.

**Omission is caught by `max_duration_s`, not by a detector.** A skipped step produces
no observation. The deadline expiring is the detection.

**Every site starts in `shadow` mode.** The engine evaluates fully and logs what it
would have done, but emits nothing. Your first-draft workflow definitions will be
wrong — durations off, legitimate variations missing. Discover that quietly, fix the
declarations, then set `enforce`.

**`confidence_components` is retained separately.** During commissioning it tells you
whether a bad event was an ambiguous detection or a good detection under poor viewing
conditions. Different fixes.

**Calibration is per site.** Camera height and lighting shift the score distribution.
`expected_calibration_error` is an ATP figure: if it is high, every threshold in the
response matrix is arbitrary.

**Nature routes to `safety_emergency`.** A rising water level is not a security incident
and should not reach the SOC as one.

**Mitigation is requested, never actuated.** iOACAM is supervisory. Safety-critical zones
are protected by independent hard safety devices. This keeps SOTIF and DO-326 scope
tractable and is worth stating explicitly on the assumptions slide.

**The policy file must be signed.** An attacker who edits it turns `violation` into
`conformant`. Sign it, anchor its version in the TPM alongside the model versions, and
stamp `policy.version` on every event the engine emits so any alert traces to the exact
policy that judged it.

## Next: build `engine/` and `harness/`

No cameras, no models. Feed scripted event sequences from a file and assert the
verdict for each of:

1. Clean run — all steps in order, within duration
2. Skipped step — s3 never arrives, deadline expires
3. Out of order — s3 before s2
4. Overrun — s2 exceeds max + tolerance
5. Underrun — s3 completes implausibly fast
6. Wrong role — maintenance_engineer performs an operator step
7. Wrong zone — correct sequence, wrong area
8. Repeated step — s3 twice
9. Unknown — an observation matching no declared step
10. Low confidence violation — correct verdict, routed to operator not alarm
11. Robot mismatch — controller claims RUNNING, vision disagrees
12. Sensor loss — `covered_by` sensors unhealthy, zone degrades to `unknown`

Cases 2, 9 and 12 are the ones that matter most and are the easiest to get wrong.
