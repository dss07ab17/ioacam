#!/usr/bin/env python3
"""Validate a JSON-lines event stream against schema/event.schema.json.

    python perception/perceive.py --max-frames 200 | python perception/tools/validate_events.py
    python perception/tools/validate_events.py events.jsonl

Uses jsonschema when it is installed and falls back to a stdlib subset checker
otherwise, so this runs in the same bare environment the engine's suites do.
The fallback covers required fields, enums, types, bounds and
additionalProperties -- which is every constraint this schema actually uses.

Beyond the schema it checks two invariants the schema cannot express:
timestamps must be non-decreasing, and a source the schema calls certain must
carry confidence exactly 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "event.schema.json"
CERTAIN_SOURCES = {"robot_bus", "machine_bus", "access_control", "timer", "engine"}

TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int,
}


def check(value, schema: dict, path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected:
        python_type = TYPES[expected]
        # bool is a subclass of int in Python; the schema means them separately.
        wrong = not isinstance(value, python_type) or (
            expected in ("integer", "number") and isinstance(value, bool)
        )
        if wrong:
            errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: {value} below minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        errors.append(f"{path}: {value} above maximum {schema['maximum']}")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")

    if "oneOf" in schema:
        matches = 0
        for branch in schema["oneOf"]:
            branch_errors: list[str] = []
            check(value, branch, path, branch_errors)
            matches += not branch_errors
        if matches != 1:
            errors.append(f"{path}: matched {matches} oneOf branches, expected exactly 1")

    if isinstance(value, dict) and "properties" in schema:
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in schema["properties"]:
                    errors.append(f"{path}: unexpected field {key!r}")
        for key, sub in value.items():
            if key in schema["properties"]:
                check(sub, schema["properties"][key], f"{path}.{key}", errors)


def validate(event: dict, schema: dict, validator) -> list[str]:
    if validator is not None:
        return [f"$.{'.'.join(str(p) for p in e.path)}: {e.message}"
                for e in validator.iter_errors(event)]
    errors: list[str] = []
    check(event, schema, "$", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="JSONL file; omit to read stdin")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        backend = "jsonschema"
    except ImportError:
        validator = None
        backend = "stdlib fallback"

    stream = open(args.path, encoding="utf-8") if args.path else sys.stdin
    total = failed = 0
    last_timestamp = -1
    seen_ids: set[str] = set()

    with stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"line {lineno}: not JSON: {exc}", file=sys.stderr)
                failed += 1
                continue

            errors = validate(event, schema, validator)

            timestamp = event.get("timestamp_us", 0)
            if timestamp < last_timestamp:
                errors.append(
                    f"timestamp_us went backwards: {timestamp} after {last_timestamp}"
                )
            last_timestamp = max(last_timestamp, timestamp)

            event_id = event.get("event_id")
            if event_id in seen_ids:
                errors.append(f"duplicate event_id {event_id!r}")
            seen_ids.add(event_id)

            if event.get("source") in CERTAIN_SOURCES and event.get("confidence") != 1.0:
                errors.append(
                    f"source {event['source']!r} is a fact but confidence is "
                    f"{event.get('confidence')}, must be exactly 1.0"
                )

            if errors:
                failed += 1
                print(f"line {lineno}: {len(errors)} problem(s)", file=sys.stderr)
                for err in errors:
                    print(f"  {err}", file=sys.stderr)

    if not args.quiet:
        print(f"{total - failed}/{total} events valid ({backend})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
