# Identity layer

Binds camera tracks to enrolled identities using what the doors already know.

```bash
python3 identity/test_identity.py
```

## Why this exists

The door device authenticates a face at 30cm, frontal, cooperative subject,
controlled light. The monitoring camera sees the same person at 8 metres,
oblique, in a helmet, possibly facing away. Those are not the same problem.

So we do not solve the hard one. The door match is carried forward, and the
workspace camera only has to answer a much easier question: *which of the few
people known to be in this zone is this?*

That reframing is what makes workspace identification viable. Matching against
four templates is a different statistical problem from matching against five
hundred, and the difference is not small.

## The cascade

| Tier | Candidates | Outcome |
|---|---|---|
| 1 | Badged into **this zone** | Expected person, expected place. Bind and carry on. |
| 2 | Everyone still **on site** | Known person in a zone they never badged into. Bind, **and raise a finding**. |
| 3 | No confident match | Unaccounted presence. Tailgating, unlogged visitor, or intruder. |

The tier that fails is more informative than the tier that succeeds.

Tier 3 is the case a door-only product is structurally blind to: it knows who
opened the door and has no idea how many people walked through. Nothing in the
market study can produce this finding.

## Thresholds are derived, not chosen

The reason a threshold must tighten as the candidate list grows is arithmetic,
not intuition. If one comparison has false match rate `p`, comparing against
`N` candidates gives roughly `N·p` chance that at least one wrong person clears
the bar. Holding a threshold fixed while `N` goes from 4 to 200 multiplies the
false match rate by fifty.

So the policy is expressed the other way round. Declare the false match rate
you accept **per decision**, and derive the per-comparison threshold:

```
required per-comparison FMR  ≤  target / N
```

With the example ROC and a target of 1e-4:

```
   4 candidates -> threshold 0.82
  20 candidates -> threshold 0.90
 200 candidates -> cannot decide
```

### The ceiling, and what it means operationally

That last line is not a bug. With a strongest measured operating point of
`p`, the arithmetic permits at most `target / p` candidates. Beyond that every
comparison is refused and **tier 2 stops working entirely**.

`ThresholdPolicy.max_candidates()` computes it. Run it at commissioning and put
the number in front of whoever owns the site, because it decides whether
whole-site matching is viable at all.

The consequence is operational: **the site roster has to be kept small.** That
is what badge-out readers, `presence_ttl_s` and the end-of-day reset are
actually for. They are not hygiene, they are what keeps tier 2 functional.

If the ceiling is below your headcount, the options are a stronger model, a
looser target, or accepting that tier 2 only works for small sites.

## Two rules that prevent naming the wrong person

**The margin rule.** The best match must beat the runner-up by `min_margin`.
Two candidates at 0.87 and 0.855 are indistinguishable; picking the higher one
attaches a name to noise, and that name may later appear on a violation.

**Match once per track, not per frame.** The answer cannot change while a track
persists, and re-running it every frame buys nothing. It also means a tracker id
swap triggers a fresh match rather than silently carrying the wrong name
forward. Call `forget(track_id)` when a track ends.

## Authentication is not attribution

The door match is an **authentication**: the person presented themselves and
the device verified them. Confidence 1.0.

The workspace match is an **attribution**: we believe this track is that person.
It carries the match score as its confidence, precisely so downstream rules can
refuse to act on a weak one. A 0.6 attribution is enough to follow a process; it
is not enough to accuse someone.

They must not be recorded the same way. If a badge is lent, the entire chain
downstream is wrong, and the log has to show which link was inferred.

## Events emitted

Three observations, distinct on purpose:

- `person_identified` — attribution, with match confidence
- `presence_unbadged` — known person, unbadged zone (tier 2)
- `identity_unverified` — nobody we can account for (tier 3)

The second and third must not be inferable merely from the absence of the
first. The engine's zone rules judge them; this layer only observes.

## Regulatory note

This cuts both ways. Fewer biometric checks in the workspace is good. But
binding a name to continuous movement through a facility is a stronger form of
worker monitoring than anonymous detection, and it is squarely what the EU AI
Act's employment provisions target. See the regulatory triage document.

## Not built

- The real `FaceMatcher`. `StubMatcher` exists for tests; the production
  implementation wraps whatever embedding model the door device already uses.
  Nothing else in the codebase knows which one.
- Face crop extraction from the frame, which belongs in `perception/`.
- Multi-camera roster sharing. Single device only.
