# `identity/data/`

The employee directory lives here, and nothing else does.

```
employees.example.csv   committed. Three fictional rows. Copy it.
employees.csv           the real one. GITIGNORED. Never commit it.
```

```bash
cp identity/data/employees.example.csv identity/data/employees.csv
```

## Columns

| Column | Required | Meaning |
|---|---|---|
| `employee_id` | yes | The badge ID, and the only key. `EMP-4471`. |
| `name` | no | For the screen. Blank is fine; the ID is shown instead. |
| `role` | no | `operator`, `maintenance_engineer`, `visitor`. See below — not cosmetic. |
| `department` | no | Display, and the first question asked about a leaver match. |
| `active` | no | `yes`/`no`. `no` means they have left. |

Extra columns are ignored, so an HR export can be used as-is. A missing
`employee_id` column means the file is ignored entirely, with one line on
stderr saying so. Rows with no ID are skipped. A repeated ID keeps the first
row and warns. Anything unparseable in `active` is treated as *unstated*, not
as a leaver — refusing to enrol a current employee over a typo is worse than
carrying an unknown.

## This file is display-only

Events, logs and the event store carry `employee_id` and never `name`. That is
the point of keying on an ID: years of event history contain no personal names
on their face, and the mapping to a person is this one small file, which can be
access-controlled, audited and deleted on its own schedule.

**Nothing downstream may require this file to exist.** Delete it and the system
runs exactly as before, showing IDs. If something breaks without it, a name has
leaked into a path that should only ever have carried an ID.

Two sanctioned non-display uses, both narrow and both documented in
[../README.md](../README.md): supplying a `role` the badge reader did not send
(off by default, `directory.supply_roles`), and refusing to enrol — or flagging
a match against — an ID marked `active=no`.

## Access control

This is the file to protect. On a device:

```bash
chmod 640 identity/data/employees.csv      # operator reads, nobody else
```

It is personal data under the GDPR and under the DPDP Act; the event store,
keyed on IDs, is a much weaker form of the same thing. Keep them separable.
