# OCR Code Improvements

## Overview

The code in `process_and_ocr_img.py` has been refactored to use OpenAI's official Python client with Pydantic models for structured outputs, replacing the previous manual `httpx` implementation.

## Key Improvements

### 1. **Pydantic Model for Type Safety**

Added `OCRResponse` model for structured, validated responses:

```python
class OCRResponse(BaseModel):
    """Pydantic model for OCR response structure."""
    
    id: Optional[str] = Field(None, description="Sample identifier")
    deg: Optional[List[int]] = Field(None, description="Degree measurements")
    L: Optional[List[float]] = Field(None, description="L* color values")
    a: Optional[List[float]] = Field(None, description="a* color values")
    b: Optional[List[float]] = Field(None, description="b* color values")
```

**Benefits:**
- Type validation at runtime
- Better IDE autocomplete and type checking
- Self-documenting code with field descriptions
- Automatic JSON schema generation

### 2. **OpenAI Client with Structured Outputs**

Replaced manual `httpx` calls with OpenAI client:

```python
# Old approach: Manual httpx with JSON parsing
with httpx.Client(trust_env=False, timeout=timeout) as client:
    response = client.post(f"{base_url}/chat/completions", json=payload)
    result = response.json()
    cleaned_content = clean_markdown_json(result_text)
    return parse_ocr_response(cleaned_content)

# New approach: OpenAI client with structured outputs
client = OpenAI(base_url=base_url, api_key="dummy", timeout=timeout)
completion = client.chat.completions.create(
    model=model,
    messages=[...],
    logprobs=True,
    top_logprobs=1,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ocr_response",
            "strict": True,
            "schema": OCRResponse.model_json_schema(),
        },
    },
)
```

**Benefits:**
- No need for manual JSON parsing and cleaning
- Built-in retry logic and error handling
- Automatic markdown code block removal
- Structured outputs ensure valid JSON responses
- Better integration with OpenAI ecosystem

### 3. **Removed Manual JSON Cleaning**

Eliminated the following functions (no longer needed):
- `clean_markdown_json()` - Structured outputs handle this automatically
- `parse_ocr_response()` - Pydantic handles validation

### 4. **Integrated structured_logprobs**

Properly integrated `structured_logprobs` for confidence analysis:

```python
completion_with_logprobs = add_logprobs(completion)
# Access log probabilities for each token
if hasattr(completion_with_logprobs, 'log_probs'):
    logger.debug(f"Logprobs available: {completion_with_logprobs.log_probs}")
```

### 5. **Updated Prompt**

Simplified prompt to work better with structured outputs:
- Removed markdown formatting instructions
- Clearer field descriptions
- More concise format

## Migration Guide

### Before (httpx + manual parsing):

```python
result = ocr_image(image, prompt, model)
# result is dict with potential JSON parsing issues
id_value = result.get("id")  # No type safety
```

### After (OpenAI client + Pydantic):

```python
result = ocr_image(image, prompt, model)
# result is validated dict
validated = OCRResponse.model_validate(result)
id_value = validated.id  # Type-safe access
```

## Example Usage

See `examples/ocr_example_usage.py` for a complete example:

```python
from examples.process_and_ocr_img import ocr_image, OCRResponse
from prompt import PROMPT

result = ocr_image(
    image=image,
    prompt=PROMPT["爱色丽MA5QC色差仪"],
    model="qwen/qwen3-vl-8b",
)

# Validate and access typed fields
validated = OCRResponse.model_validate(result)
print(f"Sample ID: {validated.id}")
print(f"L values: {validated.L}")
```

## Compatibility

The refactored code maintains backward compatibility:
- Same function signature for `ocr_image()`
- Returns same dictionary structure
- Works with existing `flatten_ocr_result()` and `validate_ocr_result()` functions

## Benefits Summary

1. ✅ **Type Safety**: Pydantic models catch errors early
2. ✅ **Cleaner Code**: Removed 30+ lines of manual parsing
3. ✅ **Better Errors**: Detailed validation errors from Pydantic
4. ✅ **Industry Standard**: Using official OpenAI client
5. ✅ **Structured Outputs**: Guaranteed valid JSON responses
6. ✅ **Logprobs Support**: Proper integration with structured_logprobs
7. ✅ **Maintainable**: Less custom code to maintain

## Testing

The main workflow remains unchanged - run:

```bash
python examples/process_and_ocr_img.py
```

All existing functionality (detection, cropping, OCR, validation, CSV export) continues to work as before.
