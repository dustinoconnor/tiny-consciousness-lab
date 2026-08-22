#!/usr/bin/env python3
"""Strict, passive storage for prospectively verified Tiny Scientist rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORMAT = "tiny_scientist_verified_production_memory_v1"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class VerifiedProductionMemory:
    """Read-only rule store. Loading or querying it cannot change control."""

    def __init__(self, path=None):
        self.path = Path(path).expanduser().resolve() if path else None
        self.rules = []
        if self.path is not None:
            self.load()

    def load(self):
        if not self.path.exists():
            raise ValueError("verified_production_memory_missing")
        payload = _load_json(self.path)
        if payload.get("format") != FORMAT:
            raise ValueError("unsupported_verified_production_memory_format")
        rules = payload.get("rules")
        if not isinstance(rules, list):
            raise ValueError("verified_production_memory_rules_missing")
        for rule in rules:
            if float(rule.get("authority", 1.0)) != 0.0:
                raise ValueError("verified_production_memory_requires_zero_authority")
            if rule.get("control_permissions") != []:
                raise ValueError("verified_production_memory_must_be_read_only")
        self.rules = rules

    def audit(self):
        return {
            "mode": "passive_verified_production_memory",
            "loaded": self.path is not None,
            "rule_count": len(self.rules),
            "authority": 0.0,
            "control_permissions": [],
            "action_influence": 0,
        }


def rule_from_verified_summary(summary, source_path):
    if summary.get("verdict") != "preregistered_verification_pass":
        raise ValueError("rule_commit_requires_preregistered_verification_pass")
    if summary.get("rule_status") != "eligible_for_verified_shadow_production_memory":
        raise ValueError("rule_commit_requires_eligible_verified_rule")
    hypothesis = summary.get("hypothesis_under_test")
    if not isinstance(hypothesis, dict):
        raise ValueError("rule_commit_missing_hypothesis")
    cause = str(hypothesis.get("cause", "")).strip().lower()
    effect = str(hypothesis.get("effect", "")).strip().lower()
    delay = float(hypothesis.get("predicted_delay_seconds", -1.0))
    if cause != "red pickup" or effect != "increased internal pressure" or delay <= 0.0:
        raise ValueError("rule_commit_unrecognized_verified_contract")
    return {
        "rule_id": "red_pickup_delayed_pressure_increase_v1",
        "cause": cause,
        "effect": effect,
        "predicted_delay_seconds": delay,
        "authority": 0.0,
        "control_permissions": [],
        "status": "verified_shadow_production_memory",
        "verification_source": str(Path(source_path)),
        "verification_date": summary.get("date"),
    }


def commit_verified_rule(summary_path, memory_path):
    rule = rule_from_verified_summary(_load_json(summary_path), summary_path)
    path = Path(memory_path).expanduser().resolve()
    if path.exists():
        memory = VerifiedProductionMemory(path)
        rules = memory.rules
    else:
        rules = []
    existing = [item for item in rules if item.get("rule_id") == rule["rule_id"]]
    if existing and existing[0] != rule:
        raise ValueError("rule_commit_conflicts_with_existing_verified_rule")
    if not existing:
        rules.append(rule)
    payload = {"format": FORMAT, "rules": rules}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return VerifiedProductionMemory(path).audit() | {"committed_rule": rule["rule_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-summary", required=True)
    parser.add_argument("--memory", required=True)
    args = parser.parse_args()
    print(json.dumps(commit_verified_rule(args.verification_summary, args.memory), indent=2))


if __name__ == "__main__":
    main()
