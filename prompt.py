PROMPT = {}

PROMPT[
    "爱色丽MA5QC色差仪"
] = """
Please extract all text from this image and return structured data.

Fields to extract:
    "id": Sample identifier (e.g., "Sample 001" or "MSZ 001 001")
    "deg": List of degree measurements, typically [15, 25, 45, 75, 110]
    "L": List of L* color values (can be negative), e.g., [86.55, 64.55, 33.08, 15.71, 9.88]
    "a": List of a* color values (can be negative), e.g., [7.03, 10.02, 12.01, 13.00, 13.99]
    "b": List of b* color values (can be negative), e.g., [-11.98, -10.17, -7.55, -6.16, -5.62]

Example output structure:
    {
        "id": "Sample 000",
        "deg": [15, 25, 45, 75, 110],
        "L": [86.55, 64.55, 33.08, 15.71, 9.88],
        "a": [7.03, 10.02, 12.01, 13.00, 13.99],
        "b": [-11.98, -10.17, -7.55, -6.16, -5.62]
    }

If a field is not clear or not present in the image, set it to null.
"""
