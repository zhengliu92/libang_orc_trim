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
