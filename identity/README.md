# Identity layer

Binds camera tracks to enrolled identities using what the doors already know.

```bash
python3 identity/test_identity.py
python3 identity/test_alignment.py
python3 identity/test_directory.py
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

## The enrolment key is a person ID

`EMP-4471`, not `Rajesh`. `identity/enroll.py` validates the format against a
configurable pattern (`face.person_id_pattern`, default `[A-Z]{2,5}-[0-9]{3,6}`)
and refuses free text that looks like a name, explaining why rather than just
declining.

Three reasons, none of them style:

- **Names are not unique.** Two people share one and the store silently merges
  two faces under a single key.
- **Names change.** Marriage, correction, transliteration. The key must not.
- **The ID is already what the badge reader emits.** Roster, door and workflow
  then speak one language with no translation step between them, and a
  translation step is exactly where the wrong person gets named.

It also means the event store holds no personal names on its face, which
materially reduces what is being retained.

A **`display_name`** may be enrolled alongside, as a separate field on the
template. It is shown on screen -- `live_recognize.py` labels its boxes with it
-- and it is never a key, never compared, and never written to an event. Events
and logs carry the ID.

For a name that stays current rather than one typed once at enrolment, see
**The employee directory** below.

**Legacy keys are not rewritten.** `load_templates` warns on stderr when a key
does not match the pattern, names it, and says how to re-enrol -- then loads it
and keeps matching. Guessing somebody's badge number would be inventing data,
and dropping their enrolment would quietly stop identifying a real person.

## The employee directory

`identity/data/employees.csv` maps an employee ID to a person, so an operator
sees a name instead of a badge number. It is the only file in the system that
does, and it does nothing else.

```
employee_id,name,role,department,active
EMP-4471,Rajesh Kumar,operator,Assembly,yes
EMP-5522,Priya Nair,maintenance_engineer,Maintenance,yes
EMP-9012,Daniel Okafor,operator,Assembly,no
```

```bash
cp identity/data/employees.example.csv identity/data/employees.csv
```

The example is committed; the real file is gitignored, because it is personal
data. `identity/data/README.md` has the column detail and the file permissions
worth setting. CSV rather than `.xlsx` because `csv` is in the standard library
and the file still opens in Excel -- the person who maintains it is in HR, not
engineering, and an `.xlsx` would mean putting openpyxl on the device to read
five columns.

### It is for display, and the boundary is the architecture

Events, logs and the event store carry `employee_id` and never `name`.

That is the entire reason for keying on an ID. Years of event history contain
no personal names on their face; the mapping to a person is this one small
file, which can be access-controlled, audited and deleted on a schedule of its
own. The event store is a much weaker form of the same personal data, and the
two stay separable only as long as nothing copies a name across.

`IdentityStage` reads the directory in exactly one method, `display_label`.
`subject_for`, which builds the `subject` block that goes into every event, does
not read it at all -- and a `name` key there would not even be schema-valid.

**Nothing may require the file to exist.** Delete it and the system runs as
before, labelling with IDs. If something breaks without it, a name has leaked
into a path that should only have carried an ID.

### What it costs to run

| Property | Behaviour |
|---|---|
| Loaded once | At startup, into a dict. A lookup is a dict hit. Never per frame, never per recognition. |
| Absence tolerated | No file, no `employee_id` column, an unreadable file: all mean no names. Never an exception. |
| Misses are quiet | An unknown ID falls back to the ID and logs **once per ID**. At 25 fps a per-frame line is 90,000 an hour. |
| Reloadable | `reload()` on demand, or `auto_reload` watching `(mtime, size)` at most every `stat_interval_s`. |

Reload matters operationally: staff change weekly, and restarting a device to
add one person is not a procedure anybody will follow. A file that appears
*after* startup is picked up too -- which is the real order of events, because
the device ships before HR sends the CSV.

The file is read defensively, because it is maintained by hand in Excel: a BOM,
padded cells, a header in the wrong case and extra columns from an HR export
all load. A row with no ID is skipped, a repeated ID keeps the first row, and
each problem is named with its line number on stderr.

### Two things it is allowed to do besides display

**Supply a missing `role`** -- off by default, `directory.supply_roles`.

Role is not cosmetic: the engine uses it for zone rules and for a step's
`actor_role`. Where a badge reader sends an ID and no role, the directory can
fill the gap when the roster is built, and `IdentityResolver.badge_event_to_roster`
takes an optional `directory` for exactly that. Two deliberate decisions are
needed -- passing the directory, and setting `supply_roles` -- because a stale
role here produces a confident `wrong_role` finding against somebody who did
nothing wrong. A role that arrives with the credential always wins; the access
control system is the system of record for who is allowed where.

**Mark a leaver.** `active=no` means the person has left, and that has two
consequences.

They are **not enrollable**. `enroll.py` refuses the ID and says why: a
template for somebody who has gone keeps matching, and every match then reads
as a current employee standing in the building. An ID that is merely *absent*
from the directory is still enrollable -- contractors, new starters and sites
with no directory are all that case, and making this file a prerequisite for
enrolment would be the worse system.

And a match against one is **not a normal recognition**. The cascade still
binds it -- turning "EMP-9012, who left in March" into "somebody we cannot
account for" would discard the one useful part of the answer -- but
`Resolution.inactive` carries an `InactiveMatch` report, `live_recognize` draws
the box amber and marks the label `[INACTIVE]`, and the stage logs it once per
ID and collects it for `drain_inactive_matches()`. It means one of two things
and the system cannot tell which: the enrolment was never revoked when they
left, or somebody else is matching against their template. Both want a human.

An unparseable `active` value is treated as *unstated*, not as a departure, and
the row is named on stderr. Refusing to enrol a current employee over a typo in
a column that did not exist last month is the worse error.

**Not built:** no `identity_inactive` observation. The event schema's
observation list is closed on purpose -- adding one commits to a detector, a
feasibility assessment and an ATP test -- so an inactive match is surfaced to
the operator and does not become an engine finding. If a site wants findings
from it, that enum entry plus an `engine.py` branch is the work, and it is a
commissioning decision rather than a side effect of adding a CSV.

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

## What else carries the identity

Those three are the events this layer *emits*. The binding it makes is also
read by the rest of the perception loop, so once a track is bound every camera
event about it carries the ID in `subject`:

| Observation | Where the subject is built |
|---|---|
| `person_in_zone`, `person_left_zone` | `perception/perceive.py` |
| `action_recognised` | `perception/actions_stage.py` |
| `action_interval` | `perception/action_intervals.py`, at close time |
| `ppe_helmet`, `ppe_vest` | `perception/ppe_stage.py` |

```json
"subject": {"class": "human", "identity": "EMP-4471", "role": "operator"}
```

Three properties hold across all of them, and each is a test:

**An unbound track carries no `identity` key.** Not `null`, not `""`, not
`"unknown"`. `IdentityStage.subject_for` omits `identity` and `role` rather
than guessing, and the engine's `wrong_role` rules depend on that: they test
for a role they can act on, and a falsy placeholder is a value they would read
as an answer. An absent key says nobody has claimed to know.

**`track_id` is never replaced by the identity.** The track id is what the
camera followed; the identity is the attribution laid over it. Keeping both is
what makes the attribution re-examinable — see "Authentication is not
attribution" above. One of them is an observation and the other is a claim.

**An ID, never a name.** The directory is read only by `display_label`, which
feeds a screen. Nothing that writes an event touches it.

Those stages reach the binding through a plain callable (`subject_for`) rather
than by importing this package, so `perception/` keeps no dependency on
`identity/` outside `perception/identity_stage.py` — which is the one module
whose job is the join.

## Regulatory note

This cuts both ways. Fewer biometric checks in the workspace is good. But
binding a name to continuous movement through a facility is a stronger form of
worker monitoring than anonymous detection, and it is squarely what the EU AI
Act's employment provisions target. See the regulatory triage document.

## Built since

- **The real `FaceMatcher`.** `arcface_matcher.py` runs ArcFace ResNet100 over
  ONNX Runtime, selected by `face.backend: "arcface-onnx"` in the identity
  config. `StubMatcher` is still what the tests use, and nothing outside
  `matcher.build_matcher` knows which one is running.
- **Face crop extraction.** `perception/face/` detects faces with RFB-320 and
  `alignment.py` warps them onto ArcFace's 112x112 template; an unaligned crop
  is refused rather than embedded, because the two are not comparable.
- **The employee directory.** `directory.py` maps IDs to names for the screen,
  reloadably, from `identity/data/employees.csv`. Display-only by
  construction; see above for the two narrow exceptions and why each is
  opt-in.
- **Binding inside the perception loop.** `perception/identity_stage.py`
  attributes each face to a person track and runs the cascade, so every event
  the loop emits for a bound track carries the identity — zone entries and
  exits, `action_recognised`, `action_interval` and the PPE pair. See "What
  else carries the identity" above. Opt-in, via `--identity` or the
  `identity.config` key.

## Not built

- Multi-camera roster sharing. Single device only.
- A measured ROC. `identity.example.json` ships a demo curve, clearly marked,
  and it must be replaced with one measured on the site's own cameras and
  enrolment data before anything is deployed.
