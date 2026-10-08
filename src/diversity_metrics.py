"""Order-invariant diversity of accepted six-slot pilot teams (no model imports).

Keys cover the pilot's categorical fields and Stat Points, not arbitrary format
fields. The caller validates teams and resolves format defaults before measuring.
"""
from collections import Counter
import hashlib
import json
import re

VERSION = "pilot-team-diversity-v1"
STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


def identifier(value):
    if not isinstance(value, str):
        raise ValueError("categorical fields must be strings")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def team_keys(team):
    """Return categorical, with-spreads and composition keys; retain multiplicity."""
    if len(team) != 6:
        raise ValueError("expected six Pokemon records")
    categorical, spreads, species = [], [], []
    for slot in team:
        try:
            # The Reg M-B validator resolves an explicitly omitted nature to
            # Serious (pokemon-showdown/sim/team-validator.ts). Missing keys
            # still fail, rather than hiding incomplete parsed records.
            names = tuple(identifier("Serious" if k == "nature" and slot[k] == "" else slot[k])
                          for k in ("species", "ability", "item", "nature"))
            moves = tuple(sorted(identifier(m) for m in slot["moves"]))
            points = tuple(slot["evs"][k] for k in STATS)
        except (KeyError, TypeError) as exc:
            raise ValueError("missing required normalized team fields") from exc
        if not names[0] or not names[1] or not names[3]:
            raise ValueError("species, ability and nature must be explicit")
        if not 1 <= len(moves) <= 4 or any(not m for m in moves):
            raise ValueError("expected one to four explicit moves")
        if any(type(x) is not int or x < 0 for x in points):
            raise ValueError("Stat Points must be nonnegative integers")
        # Padding represents absent move slots, not unknown moves.
        record = names + moves + ("",) * (4 - len(moves))
        categorical.append(record)
        spreads.append(record + points)
        species.append(names[0])
    return tuple(sorted(categorical)), tuple(sorted(spreads)), tuple(sorted(species))


def _sha(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


class Reference:
    """Snapshot a fixed reference, labeled as training data only when verified."""
    def __init__(self, teams, *, scope, format_id, provenance=None):
        if scope not in ("initial_training", "corpus", "generation_training") or not format_id:
            raise ValueError("explicit reference scope and format required")
        keys = [team_keys(t) for t in teams]
        if not keys:
            raise ValueError("novelty needs a nonempty reference")
        self.snapshot = sorted(row[1] for row in keys)
        self.keys = tuple(frozenset(row[i] for row in keys) for i in range(3))
        self.metadata = dict(scope=scope, format=format_id, key_version=VERSION,
                             n=len(keys), unique_48=len(self.keys[0]),
                             unique_with_spreads=len(self.keys[1]),
                             content_sha256=_sha(self.snapshot),
                             provenance=provenance or {})


def _counts(keys, reference):
    n = len(keys)
    counts = Counter(keys)
    unique = len(counts)
    novel_draws = sum(c for key, c in counts.items() if key not in reference) if reference is not None else None
    novel_unique = sum(key not in reference for key in counts) if reference is not None else None
    return dict(n=n, unique_count=unique, repeat_count=n - unique,
                unique_fraction=unique / n if n else None,
                repeat_fraction=(n - unique) / n if n else None,
                novel_draw_count=novel_draws,
                novel_draw_fraction=novel_draws / n if n and reference is not None else None,
                novel_unique_count=novel_unique,
                novel_unique_fraction=novel_unique / unique if unique and reference is not None else None)


def measure(accepted_teams, reference=None, *, expected_n=None, attempts=None):
    """Measure accepted draws before ranking, retaining all repeated draws.

    Unknown attempt counts remain null. This function does not assert legality.
    """
    rows = [team_keys(t) for t in accepted_teams]
    n = len(rows)
    if expected_n is not None and (type(expected_n) is not int or expected_n < 1):
        raise ValueError("expected_n must be a positive integer")
    if attempts is not None and (type(attempts) is not int or attempts < n):
        raise ValueError("attempts must be an integer at least the accepted count")
    counts = Counter(row[2] for row in rows)
    metrics = [_counts([row[i] for row in rows], reference.keys[i] if reference else None)
               for i in range(3)]
    metrics[2].update(frequencies=[dict(species=list(key), count=count)
                                    for key, count in sorted(counts.items())],
                      largest_share=max(counts.values()) / n if n else None)
    return dict(key_version=VERSION,
                status="empty" if not n else "sample_size_mismatch" if expected_n is not None and n != expected_n else "ok",
                sampling=dict(population="validator_accepted_before_ranking", accepted=n,
                              expected_n=expected_n, attempts=attempts,
                              rejected=attempts - n if attempts is not None else None,
                              acceptance_fraction=n / attempts if attempts else None,
                              duplicates_retained=True),
                reference=reference.metadata if reference else None,
                categorical_48=metrics[0], with_spreads=metrics[1], composition=metrics[2])
