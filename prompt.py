PROMPT = {}

PROMPT[
    "爱色丽MA5QC色差仪"
] = """
Please extract all text from this image. Return only the extracted text without any additional explanation. 

Fields example:
    "id": str, example: "Sample 001" or "MSZ 001 001",
    "deg": List[int], always return [15, 25, 45, 75, 110],
    "L": List[float], example: [86.55, 64.55, 33.08, 15.71, 9.88], NOTE: value can be negative.
    "a": List[float], example: [7.03, 10.02, 12.01, 13.00, 13.99], NOTE: value can be negative.
    "b": List[float], example: [-11.98, -10.17, -7.55, -6.16, -5.62], NOTE: value can be negative.
    
Output the data in the following JSON format:
    {
        "id": "Sample 000",
        "deg": [15, 25, 45, 75, 110],
        "L": [86.55, 64.55, 33.08, 15.71, 9.88],
        "a": [7.03, 10.02, 12.01, 13.00, 13.99],
        "b": [-11.98, -10.17, -7.55, -6.16, -5.62],
    }
If the field is not clear to you, leave it as NULL.
"""


PROMPT[
    "score"
] = """
Evaluate the quality and confidence of the data extraction from this image.

Consider the following factors:
- Text clarity and readability (blurriness, contrast, resolution)
- Completeness of data (all expected fields are present and extractable)
- Numerical accuracy (numbers are clear and unambiguous)
- Image quality (lighting, distortion, occlusion)
- Field alignment and structure consistency

Provide a confidence score between 0 and 1:
- 0.9-1.0: Excellent - All text is crystal clear, all fields are complete, no ambiguity
- 0.7-0.9: Good - Text is readable, most fields are clear with minor uncertainties
- 0.5-0.7: Fair - Some text is unclear or missing, requires manual verification
- 0.3-0.5: Poor - Significant portions unclear or missing, high uncertainty
- 0.0-0.3: Very Poor - Most data is unreadable or missing

Return only a single number (e.g., 0.85) without explanation.
"""
