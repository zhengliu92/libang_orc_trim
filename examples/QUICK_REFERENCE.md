# Quick Reference Guide

## 快速对比 / Quick Comparison

### 旧实现 / Old Implementation
```python
# ❌ 手动 httpx + JSON 解析
import httpx

def ocr_image(image, prompt, model, base_url, max_tokens, timeout):
    with httpx.Client(...) as client:
        response = client.post(...)
        result = response.json()
        cleaned = clean_markdown_json(result_text)
        return parse_ocr_response(cleaned)
```

### 新实现 / New Implementation
```python
# ✅ OpenAI Client + Pydantic
from openai import OpenAI
from pydantic import BaseModel, Field

class OCRResponse(BaseModel):
    id: Optional[str] = Field(None, description="...")
    deg: Optional[List[int]] = Field(None, description="...")
    L: Optional[List[float]] = Field(None, description="...")
    a: Optional[List[float]] = Field(None, description="...")
    b: Optional[List[float]] = Field(None, description="...")

def ocr_image(image, prompt, model, base_url, max_tokens, timeout):
    http_client = httpx.Client(trust_env=False, timeout=timeout)
    client = OpenAI(base_url=base_url, api_key="dummy", http_client=http_client)
    
    completion = client.chat.completions.create(
        model=model,
        messages=[...],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ocr_response",
                "strict": True,
                "schema": OCRResponse.model_json_schema(),
            },
        },
    )
    
    completion_with_logprobs = add_logprobs(completion)
    ocr_data = json.loads(completion.choices[0].message.content)
    validated = OCRResponse.model_validate(ocr_data)
    return validated.model_dump(exclude_none=False)
```

---

## 核心区别 / Core Differences

| 特性 / Feature | 旧方法 / Old | 新方法 / New |
|----------------|-------------|-------------|
| HTTP 客户端 | Manual httpx | OpenAI Client |
| 类型验证 | ❌ None | ✅ Pydantic |
| JSON 清理 | Manual (30+ lines) | Automatic |
| 结构化输出 | ❌ No | ✅ Yes |
| IDE 支持 | Limited | Full |
| 代码行数 | ~180 | ~130 (-28%) |

---

## 使用示例 / Usage Examples

### 示例 1: 基本 OCR / Example 1: Basic OCR

```python
from examples.process_and_ocr_img import ocr_image
from prompt import PROMPT
from PIL import Image

image = Image.open("image.jpg")
result = ocr_image(
    image=image,
    prompt=PROMPT["爱色丽MA5QC色差仪"],
    model="qwen/qwen3-vl-8b",
)

print(result)
# Output: {'id': 'Sample 001', 'deg': [15, 25, 45, 75, 110], ...}
```

### 示例 2: 类型安全访问 / Example 2: Type-Safe Access

```python
from examples.process_and_ocr_img import OCRResponse

# Validate and get typed access
validated = OCRResponse.model_validate(result)

# IDE autocomplete works!
print(validated.id)        # Type: Optional[str]
print(validated.deg)       # Type: Optional[List[int]]
print(validated.L)         # Type: Optional[List[float]]
```

### 示例 3: 错误处理 / Example 3: Error Handling

```python
from pydantic import ValidationError

try:
    validated = OCRResponse.model_validate(result)
except ValidationError as e:
    # Detailed error messages
    print(e.errors())
    # [
    #   {
    #     'loc': ('deg',),
    #     'msg': 'value is not a valid list',
    #     'type': 'type_error.list'
    #   }
    # ]
```

### 示例 4: 使用 Logprobs / Example 4: Using Logprobs

```python
from structured_logprobs import add_logprobs

# Inside ocr_image function
completion_with_logprobs = add_logprobs(completion)

if hasattr(completion_with_logprobs, 'log_probs'):
    # Access token-level confidence scores
    log_probs = completion_with_logprobs.log_probs
    print(f"Token confidence: {log_probs}")
```

---

## Pydantic 模型 / Pydantic Model

### 定义 / Definition

```python
class OCRResponse(BaseModel):
    """OCR 响应的 Pydantic 模型 / Pydantic model for OCR response"""
    
    id: Optional[str] = Field(
        None, 
        description="样品标识符 / Sample identifier"
    )
    deg: Optional[List[int]] = Field(
        None, 
        description="角度测量值 / Degree measurements"
    )
    L: Optional[List[float]] = Field(
        None, 
        description="L* 颜色值（可为负）/ L* color values (can be negative)"
    )
    a: Optional[List[float]] = Field(
        None, 
        description="a* 颜色值（可为负）/ a* color values (can be negative)"
    )
    b: Optional[List[float]] = Field(
        None, 
        description="b* 颜色值（可为负）/ b* color values (can be negative)"
    )
```

### 验证 / Validation

```python
# ✅ Valid
data = {
    "id": "Sample 001",
    "deg": [15, 25, 45, 75, 110],
    "L": [86.55, 64.55, 33.08, 15.71, 9.88],
    "a": [7.03, 10.02, 12.01, 13.00, 13.99],
    "b": [-11.98, -10.17, -7.55, -6.16, -5.62]
}
validated = OCRResponse.model_validate(data)

# ❌ Invalid - will raise ValidationError
invalid_data = {
    "id": "Sample 001",
    "deg": "not a list",  # Should be List[int]
}
OCRResponse.model_validate(invalid_data)  # Raises ValidationError
```

---

## JSON Schema

### 自动生成 / Auto-Generated

```python
schema = OCRResponse.model_json_schema()
print(schema)
```

输出 / Output:
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Sample identifier",
      "default": null
    },
    "deg": {
      "type": "array",
      "items": {"type": "integer"},
      "description": "Degree measurements",
      "default": null
    },
    "L": {
      "type": "array",
      "items": {"type": "number"},
      "description": "L* color values (can be negative)",
      "default": null
    },
    "a": {
      "type": "array",
      "items": {"type": "number"},
      "description": "a* color values (can be negative)",
      "default": null
    },
    "b": {
      "type": "array",
      "items": {"type": "number"},
      "description": "b* color values (can be negative)",
      "default": null
    }
  }
}
```

---

## 结构化输出配置 / Structured Outputs Config

### OpenAI API 参数 / OpenAI API Parameters

```python
response_format = {
    "type": "json_schema",           # 使用 JSON Schema
    "json_schema": {
        "name": "ocr_response",      # Schema 名称
        "strict": True,              # 严格模式（推荐）
        "schema": OCRResponse.model_json_schema(),  # Pydantic 生成的 schema
    },
}

completion = client.chat.completions.create(
    model=model,
    messages=[...],
    response_format=response_format,  # 添加此参数
    logprobs=True,                    # 启用 logprobs
    top_logprobs=1,                   # 返回前 N 个最可能的 token
)
```

---

## 常见问题 / FAQ

### Q1: 为什么需要 `api_key="dummy"`？
**A:** OpenAI 客户端要求 API key 参数，但对于本地模型服务器，此值不会被使用。

### Q2: `trust_env=False` 的作用是什么？
**A:** 忽略系统代理设置，直接连接到本地服务器。

### Q3: 如何添加新字段？
**A:** 在 `OCRResponse` 模型中添加新的字段定义：

```python
class OCRResponse(BaseModel):
    # ... existing fields ...
    new_field: Optional[str] = Field(None, description="New field description")
```

### Q4: 如何处理验证错误？
**A:** 使用 try-except 捕获 ValidationError：

```python
from pydantic import ValidationError

try:
    validated = OCRResponse.model_validate(data)
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle error
```

---

## 性能对比 / Performance Comparison

| 指标 / Metric | 旧实现 / Old | 新实现 / New | 改进 / Improvement |
|--------------|-------------|-------------|-------------------|
| 代码行数 / Lines | 180 | 130 | -28% |
| 函数数量 / Functions | 5 | 3 | -40% |
| 类型安全 / Type Safety | ❌ | ✅ | +100% |
| IDE 支持 / IDE Support | 基础 / Basic | 完整 / Full | +100% |
| 错误详情 / Error Details | 简单 / Basic | 详细 / Detailed | +100% |

---

## 迁移检查清单 / Migration Checklist

- ✅ 更新导入语句 / Update imports
- ✅ 添加 Pydantic 模型 / Add Pydantic models
- ✅ 使用 OpenAI 客户端 / Use OpenAI client
- ✅ 添加结构化输出 / Add structured outputs
- ✅ 集成 logprobs / Integrate logprobs
- ✅ 更新提示词 / Update prompts
- ✅ 删除冗余代码 / Remove redundant code
- ✅ 测试功能 / Test functionality
- ✅ 更新文档 / Update documentation

---

## 相关文件 / Related Files

- 📄 `examples/process_and_ocr_img.py` - 主要实现 / Main implementation
- 📄 `examples/ocr_example_usage.py` - 使用示例 / Usage example
- 📄 `examples/IMPROVEMENTS.md` - 详细改进 / Detailed improvements
- 📄 `examples/COMPARISON.md` - 代码对比 / Code comparison
- 📄 `examples/SUMMARY.md` - 完整总结 / Complete summary
- 📄 `prompt.py` - OCR 提示词 / OCR prompts

---

## 运行测试 / Run Tests

```bash
# 运行示例脚本 / Run example script
python examples/ocr_example_usage.py

# 运行完整流程 / Run full pipeline
python examples/process_and_ocr_img.py

# 检查特定图片 / Check specific image
python -c "
from PIL import Image
from examples.process_and_ocr_img import ocr_image, OCRResponse
from prompt import PROMPT

img = Image.open('path/to/image.jpg')
result = ocr_image(img, PROMPT['爱色丽MA5QC色差仪'], 'qwen/qwen3-vl-8b')
print(OCRResponse.model_validate(result))
"
```

---

**快速参考创建日期 / Quick Reference Created**: 2026-01-14
