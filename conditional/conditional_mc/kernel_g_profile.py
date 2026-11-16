"""Explicit, reproducible location/sample-size policies for kernel-g inference.

The profile never reads a simulated error label or truth. Any skewness gate
is evaluated from the training residuals by kernel_g.prepare_g_fits.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


def validate_profile(profile):
    from .kernel_g import validate_options
    if profile is None:
        return None
    out = deepcopy(profile)
    if not isinstance(out, dict):
        raise ValueError('kernel-g profile must be an object')
    unknown = set(out) - {'version', 'name', 'methods', 'rules', 'notes'}
    if unknown:
        raise ValueError(f'unknown kernel-g profile fields: {sorted(unknown)}')
    if out.get('version') != 1 or not isinstance(out.get('name'), str) or not out['name']:
        raise ValueError('kernel-g profile needs version=1 and a nonempty name')
    if not out.get('methods') or not set(out['methods']) <= {'semi', 'conf', 'conf_hsemi'}:
        raise ValueError('invalid kernel-g methods')
    if not isinstance(out.get('rules'), list):
        raise ValueError('kernel-g rules must be a list')
    for rule in out['rules']:
        if not isinstance(rule, dict) or set(rule) - {'locations', 'n_min', 'n_max', 'options'}:
            raise ValueError('invalid kernel-g rule fields')
        locations = rule.get('locations')
        if not isinstance(locations, list) or not locations or any(
                not math.isfinite(float(s)) or abs(float(s)) >= 1 for s in locations):
            raise ValueError('kernel-g rule needs finite interior locations')
        lower, upper = int(rule.get('n_min', 1)), int(rule.get('n_max', 2**63-1))
        if lower < 1 or upper < lower:
            raise ValueError('invalid kernel-g sample-size range')
        rule['options'] = validate_options(rule.get('options', {}))
    return out


def resolve_g_options(profile, method, n, s0):
    if profile is None or method not in profile['methods']:
        return None
    matches = [r for r in profile['rules']
               if int(r.get('n_min', 1)) <= int(n) <= int(r.get('n_max', 2**63-1))
               and any(math.isclose(float(s0), float(s), rel_tol=0, abs_tol=1e-10)
                       for s in r['locations'])]
    if len(matches) > 1:
        raise ValueError('overlapping kernel-g profile rules')
    return deepcopy(matches[0]['options']) if matches else None


def profile_id(profile):
    payload = json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
