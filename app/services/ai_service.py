import base64
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import HTTPException
import os

load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def analyze_face_image(image_bytes: bytes):
    """
    Sends image to OpenAI Vision model to extract skin details.
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """
    You are a professional beauty consultant and dermatologist. 
    Analyze the person's face in this image. 
    Identify their Skin Tone (Fair, Light, Medium, Tan, Deep), 
    Undertone (Cool, Neutral, Warm), Face Shape, and Eye Color.
    
    Return ONLY a JSON object:
    {
        "skin_tone": "string",
        "undertone": "string",
        "face_shape": "string",
        "eye_color": "string",
        "confidence_score": integer (0-100),
        "summary": "Short description of their complexion"
    }
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a beauty analysis AI. Output JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis failed")

async def get_shade_recommendation(profile, products):
    """
    Matches the user profile against a list of products.
    """
    prompt = f"""
    Act as a Color Theory Expert. 
    User Profile: Tone={profile.skin_tone}, Undertone={profile.undertone}.
    
    Analyze these products and find the ONE best match for this user:
    {json.dumps([p.dict() for p in products])}
    
    Return ONLY JSON:
    {{
        "best_match_id": "product_id",
        "match_score": integer (0-100),
        "reasoning": "Why this shade works for their undertone."
    }}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a color match expert. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Recommendation failed")