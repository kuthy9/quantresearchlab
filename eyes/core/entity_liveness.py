"""Which entities the Eye still retains, read off the states it publishes.

The emitter remembers, per swing, level, BOS, displacement, zone or range,
the event that last spoke about it so a later event can cite it.  Such a
memory is read for as long as some tracker or reducer retains the entity --
or retains another entity that cites it: a BOS names its target swing, an
FVG its source candles, a structural leg its ATR bars.  The harvest below
walks every retained state and collects every identifier field, so the
emitter can drop the entries no retained state can reach any more.

An identifier is any string held by a field whose name ends in ``_id`` or
``_ids`` (``item_id``, ``candidate_id``, ``source_candle_ids`` ...).  A
namespaced identity (``swing:…``, ``pool:…``) is collected in both forms,
since the memories key some entries by the bare id and some by the
namespaced one.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass


def _is_identifier_field(name: str) -> bool:
    return name.endswith("_id") or name.endswith("_ids")


def _collect(value: object, identifier: bool, out: set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if identifier:
            out.add(value)
            namespace, separator, bare = value.partition(":")
            if separator and bare and namespace.isalpha():
                out.add(bare)
        return
    if isinstance(value, (int, float, bool, bytes)):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(
                getattr(value, field.name),
                _is_identifier_field(field.name),
                out,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect(
                item,
                identifier or (isinstance(key, str) and _is_identifier_field(key)),
                out,
            )
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect(item, identifier, out)
        return
    if hasattr(value, "isoformat") or hasattr(value, "value"):
        # Timestamps and enums carry no identity.
        return


def harvest_entity_ids(*sources: object) -> frozenset[str]:
    """Every identifier any of ``sources`` cites, namespaced and bare."""

    out: set[str] = set()
    for source in sources:
        _collect(source, False, out)
    return frozenset(out)
