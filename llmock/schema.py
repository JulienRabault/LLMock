"""Deterministic example values that satisfy a JSON Schema.

Used to fill tool-call arguments and structured outputs, so that a client's
own validation -- pydantic, zod, jsonschema -- accepts what LLMock returns.
Covers the subset of JSON Schema that tool definitions use in practice:
types and type unions, enum, const, default, examples, required, nested
objects and arrays, anyOf/oneOf/allOf, local ``$ref`` into ``$defs`` or
``definitions``, string formats, and numeric and length bounds.
"""

from __future__ import annotations

from typing import Any

__all__ = ["example_for", "from_openapi"]

_MAX_DEPTH = 8

_FORMATS = {
    "date": "2026-01-01",
    "date-time": "2026-01-01T12:00:00Z",
    "time": "12:00:00",
    "email": "user@example.com",
    "uri": "https://example.com",
    "url": "https://example.com",
    "uuid": "00000000-0000-4000-8000-000000000000",
    "ipv4": "192.0.2.1",
    "ipv6": "2001:db8::1",
    "hostname": "example.com",
}


def example_for(schema: Any, *, name: str = "value") -> Any:
    """Return a value valid against ``schema``.

    ``name`` is the property the value is for; it only flavours strings, so
    that ``{"city": "mock-city"}`` reads better than ``{"city": "string"}``.
    """
    root = schema if isinstance(schema, dict) else {}
    return _example(root, root, name, 0)


def _example(schema: Any, root: dict[str, Any], name: str, depth: int) -> Any:
    if not isinstance(schema, dict) or depth > _MAX_DEPTH:
        return None

    if "$ref" in schema:
        return _example(_resolve(schema["$ref"], root), root, name, depth + 1)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        return examples[0]
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            non_null = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
            return _example((non_null or options)[0], root, name, depth + 1)
    if isinstance(schema.get("allOf"), list):
        return _example(_merge_all_of(schema), root, name, depth + 1)

    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [k for k in kind if k != "null"]
        kind = kinds[0] if kinds else "null"
    if kind is None:
        kind = "object" if "properties" in schema else "string"

    if kind == "object":
        return _object(schema, root, depth)
    if kind == "array":
        return _array(schema, root, name, depth)
    if kind == "string":
        return _string(schema, name)
    if kind == "integer":
        return int(_number(schema, integer=True))
    if kind == "number":
        return _number(schema, integer=False)
    if kind == "boolean":
        return True
    return None


def _object(schema: dict[str, Any], root: dict[str, Any], depth: int) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    return {
        key: _example(sub, root, key, depth + 1)
        for key, sub in properties.items()
        if isinstance(sub, dict)
    }


def _array(schema: dict[str, Any], root: dict[str, Any], name: str, depth: int) -> list[Any]:
    count = max(1, int(schema.get("minItems", 1)))
    if "maxItems" in schema:
        count = min(count, int(schema["maxItems"]))
    items = schema.get("items")
    if isinstance(items, list):  # tuple validation
        return [_example(sub, root, name, depth + 1) for sub in items]
    return [_example(items or {}, root, name, depth + 1) for _ in range(count)]


def _string(schema: dict[str, Any], name: str) -> str:
    value = _FORMATS.get(schema.get("format", ""), f"mock-{name}")
    min_length = int(schema.get("minLength", 0))
    if len(value) < min_length:
        value = value + "x" * (min_length - len(value))
    if "maxLength" in schema:
        value = value[: int(schema["maxLength"])]
    return value


def _number(schema: dict[str, Any], *, integer: bool) -> float:
    low = schema.get("minimum", schema.get("exclusiveMinimum"))
    high = schema.get("maximum", schema.get("exclusiveMaximum"))
    step = 1 if integer else 0.5
    if low is not None:
        value = low + (step if "exclusiveMinimum" in schema and "minimum" not in schema else 0)
    elif high is not None:
        value = high - (step if "exclusiveMaximum" in schema and "maximum" not in schema else 0)
        value = min(value, 1)
    else:
        value = 1
    if high is not None:
        ceiling = high - (step if "exclusiveMaximum" in schema and "maximum" not in schema else 0)
        value = min(value, ceiling)
    multiple = schema.get("multipleOf")
    if multiple:
        value = multiple * max(1, round(value / multiple))
    return int(value) if integer else float(value)


# Gemini's OpenAPI dialect, as google-genai actually sends it: snake_case
# keywords and upper-case type names.
_OPENAPI_KEYWORDS = {
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_items": "minItems",
    "max_items": "maxItems",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
    "any_of": "anyOf",
    "one_of": "oneOf",
    "all_of": "allOf",
    "additional_properties": "additionalProperties",
    "property_ordering": "propertyOrdering",
}
_SUBSCHEMA = frozenset({"items", "not", "additionalProperties"})
_SUBSCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SUBSCHEMA_MAPS = frozenset({"properties", "$defs", "definitions"})


def from_openapi(schema: Any) -> Any:
    """Turn a Gemini / OpenAPI-style schema into standard JSON Schema.

    Renames snake_case keywords and lower-cases type names -- but never the
    keys *inside* ``properties``: those are the user's field names.
    """
    if isinstance(schema, list):
        return [from_openapi(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for raw_key, value in schema.items():
        key = _OPENAPI_KEYWORDS.get(raw_key, raw_key)
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: from_openapi(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMA_LISTS or key in _SUBSCHEMA:
            out[key] = from_openapi(value)
        elif key == "type" and isinstance(value, str):
            out[key] = value.lower()
        else:
            out[key] = value
    return out


def _resolve(ref: Any, root: dict[str, Any]) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return {}
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node


def _merge_all_of(schema: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
    for part in schema["allOf"]:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if key == "properties":
                merged.setdefault("properties", {}).update(value)
            elif key == "required":
                merged["required"] = sorted({*merged.get("required", []), *value})
            else:
                merged.setdefault(key, value)
    return merged
