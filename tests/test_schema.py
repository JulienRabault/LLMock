"""Generated examples must pass real validators, not just look plausible."""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from llmock.schema import example_for

jsonschema = pytest.importorskip("jsonschema")


def valid(schema):
    value = example_for(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    ).validate(value)
    return value


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": "integer", "minimum": 5, "maximum": 9},
        {"type": "integer", "exclusiveMinimum": 5},
        {"type": "number", "exclusiveMaximum": 0},
        {"type": "number", "minimum": 0.25, "multipleOf": 0.25},
        {"type": "integer", "multipleOf": 7},
        {"type": "boolean"},
        {"type": "null"},
        {"type": ["string", "null"]},
        {"type": "string", "minLength": 20},
        {"type": "string", "maxLength": 3},
        {"type": "string", "format": "email"},
        {"type": "string", "format": "date-time"},
        {"type": "string", "format": "uuid"},
        {"type": "string", "format": "date"},
        {"enum": ["celsius", "fahrenheit"]},
        {"const": 42},
        {"type": "array", "items": {"type": "integer"}, "minItems": 3},
        {"type": "array", "items": {"type": "string"}, "maxItems": 0},
        {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        {"oneOf": [{"type": "string", "format": "uri"}, {"type": "integer"}]},
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"enum": ["c", "f"]},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
            },
            "required": ["city", "unit", "days"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"point": {"$ref": "#/$defs/Point"}},
            "required": ["point"],
            "$defs": {
                "Point": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                }
            },
        },
        {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
            ]
        },
    ],
)
def test_examples_validate(schema):
    valid(schema)


def test_strings_are_flavoured_by_their_property_name():
    value = example_for({"type": "object", "properties": {"city": {"type": "string"}}})
    assert value == {"city": "mock-city"}


def test_examples_and_defaults_win():
    assert example_for({"type": "string", "examples": ["Paris"]}) == "Paris"
    assert example_for({"type": "integer", "default": 3}) == 3


def test_output_is_deterministic():
    schema = {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}
    assert example_for(schema) == example_for(schema)


def test_recursive_schema_terminates():
    schema = {"$ref": "#/$defs/Node", "$defs": {"Node": {
        "type": "object",
        "properties": {"child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}},
    }}}
    example_for(schema)  # must not recurse forever


def test_garbage_schemas_do_not_crash():
    for schema in (None, "string", 42, [], {"$ref": "http://elsewhere"}, {"$ref": "#/nowhere"}):
        example_for(schema)


# -- pydantic: how agent frameworks define tools ------------------------------


class Unit(str, enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class Location(BaseModel):
    city: str
    country: str = Field(min_length=2, max_length=2)


class Forecast(BaseModel):
    location: Location
    unit: Unit
    days: int = Field(ge=1, le=14)
    when: dt.date
    include_hourly: bool
    request_id: uuid.UUID
    tags: list[str] = Field(min_length=1)
    mode: Literal["fast", "accurate"]
    note: Optional[str] = None
    threshold: Union[int, float]


def test_pydantic_models_accept_the_generated_arguments():
    arguments = example_for(Forecast.model_json_schema())
    parsed = Forecast.model_validate(arguments)
    assert parsed.location.country and len(parsed.location.country) == 2
    assert 1 <= parsed.days <= 14


# -- Gemini's OpenAPI dialect, as google-genai really sends it -----------------

from llmock.schema import from_openapi  # noqa: E402


def test_openapi_snake_case_constraints_are_honoured():
    sent_by_google_genai = {
        "type": "OBJECT",
        "properties": {
            "currency": {"type": "STRING", "min_length": 3, "max_length": 3},
            "lines": {"type": "ARRAY", "items": {"type": "STRING"}, "min_items": 1},
        },
        "required": ["currency", "lines"],
        "property_ordering": ["currency", "lines"],
    }
    value = valid(from_openapi(sent_by_google_genai))
    assert len(value["currency"]) == 3


def test_property_names_are_never_renamed():
    schema = from_openapi({"type": "OBJECT", "properties": {"min_length": {"type": "INTEGER"}}})
    assert list(schema["properties"]) == ["min_length"]
    assert schema["properties"]["min_length"]["type"] == "integer"
