import json
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator, SchemaError


def parse_response_schema(raw: Optional[str], response_format: str) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    message = 'response_schema must be an inline JSON Schema object for a JSON response (max 20000 characters)'
    if response_format == 'text' or len(raw) > 20_000:
        raise ValueError(message)
    try:
        schema = json.loads(raw)
        if not isinstance(schema, dict) or schema.get('type') != 'object' or not isinstance(schema.get('properties'), dict):
            raise ValueError(message)
        pending = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                # Forward self-contained schemas without resolving any references in either service.
                if node.keys() & {'$ref', '$dynamicRef', '$recursiveRef'}:
                    raise ValueError(message)
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
        Draft202012Validator.check_schema(schema)
    except (SchemaError, ValueError, RecursionError) as exc:
        raise ValueError(message) from exc
    return schema
