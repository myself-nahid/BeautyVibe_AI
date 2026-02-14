import base64
import json
import os
from openai import AsyncOpenAI
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY is missing in .env file!")

client = AsyncOpenAI(api_key=api_key)

async def analyze_face_image(image_bytes: bytes):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """
    You are a beauty consultant. Analyze this face.
    
    REQUIRED JSON OUTPUT FIELDS:
    1. skin_tone: [Fair, Light, Medium, Tan, Deep]
    2. undertone: [Cool, Neutral, Warm]
    3. face_shape: [Oval, Round, Square, Heart, Diamond, Oblong] (Pick the closest match, do not return null)
    4. eye_color: String
    5. summary: A short sentence describing the features.
    
    Return strictly valid JSON.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a beauty AI. Output JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        if result.get("face_shape") is None:
            summary_text = result.get("summary", "").lower()
            
            if "oval" in summary_text:
                result["face_shape"] = "Oval"
            elif "round" in summary_text:
                result["face_shape"] = "Round"
            elif "square" in summary_text:
                result["face_shape"] = "Square"
            elif "heart" in summary_text:
                result["face_shape"] = "Heart"
            elif "diamond" in summary_text:
                result["face_shape"] = "Diamond"
            elif "oblong" in summary_text:
                result["face_shape"] = "Oblong"
            else:
                result["face_shape"] = "Oval" 
        
        result["face_shape"] = result["face_shape"].capitalize()

        return result

    except Exception as e:
        print(f"AI ERROR: {str(e)}")
        return {
            "skin_tone": "Medium",
            "undertone": "Neutral",
            "face_shape": "Oval",
            "eye_color": "Brown",
            "confidence_score": 0,
            "summary": f"Error: {str(e)}"
        }

async def get_shade_recommendation(profile, products):
    """
    Matches the user profile against a list of products using AI logic.
    """
    
    products_json = json.dumps([p.dict() for p in products])
    
    prompt = f"""
    Act as a Professional Color Theorist and Makeup Artist.
    
    **User Profile:**
    - Skin Tone: {profile.skin_tone}
    - Undertone: {profile.undertone}
    - Eye Color: {profile.eye_color}
    
    **Available Products:**
    {products_json}
    
    **Task:**
    Select the SINGLE best matching product from the list that complements the user's skin tone and undertone.
    
    **Return strictly valid JSON:**
    {{
        "best_match_id": "product_id_from_list",
        "match_score": integer (0-100),
        "reasoning": "A short explanation of why this shade works best."
    }}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a color match expert. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"RECOMMENDATION API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation Failed: {str(e)}")