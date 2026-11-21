import time
import requests
import json

class GlossToArabicEngine:
    def __init__(self, api_key: str = "api here"):
        """
        Initializes the reverse translation engine with AI Error Correction and Dual JSON Output.
        """
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
        
        # =====================================================================
        # SYSTEM PROMPT: THE REVERSE INTERPRETER & JSON CORRECTOR
        # =====================================================================
        self.system_prompt = """You are an expert Egyptian Sign Language (EgySL) interpreter. 

1. RESTORE NATURAL ORDER & GRAMMAR: Convert the raw gloss into standard Egyptian Arabic flow. Connect ideas logically by adding necessary conjunctions (عشان، و، بس، عشان كده) and prepositions (في، من). Conjugate verbs naturally based on the subject and tense.
2. SIGN LANGUAGE NEGATION: In EgySL, the negation token ("لا") always appears AFTER the verb or action it negates. Apply this negation to the preceding action (e.g., action + لا = مش + action) instead of letting it bleed into subsequent words.
3. SIGN LANGUAGE QUESTIONS: Question words (like كام، فين، ليه، إمتى) are placed at the very END of the sequence in EgySL. Move them to their natural, grammatically correct position in spoken Egyptian Arabic.
4. EGYPTIAN IDIOMS & CONTEXT: Translate vocabulary into functional colloquial Egyptian Arabic based on the environment implied by the surrounding words (e.g., educational, medical, or vocational contexts). 
5. THE 2 OPTIONS & FAULT-TOLERANCE: When words completely clash semantically, the vision sensor misclassified a sign. You MUST generate two distinct logical paths:
   - Option 1 (primary): Choose the dominant contextual theme, completely drop or replace the clashing word to make a flawless sentence, and prioritize this for speech.
   - Option 2 (alternative): Formulate an alternative logical interpretation or a second valid way to link the ambiguous grammar.
6. DIALECT: 100% natural, spoken Egyptian colloquial Arabic (العامية المصرية).

FINAL OUTPUT RULE
Output ONLY a valid JSON object. No markdown tags, no explanations.
Format:
{
  "primary": "...",
  "alternative": "..."
}
"""

    def translate_to_arabic(self, gloss_text: str, max_retries: int = 4) -> str:
        if not gloss_text or not gloss_text.strip():
            return ""

        payload = {
            "system_instruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"parts": [{"text": gloss_text.strip()}]}],
            "generationConfig": {
                "temperature": 0.4, 
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json"  # Forces the API to return valid JSON
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 429:
                    print(f"⚠️ [429 Error]: Rate limit hit. Retrying...")
                    time.sleep((2 ** attempt) + 3)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} Failed: {str(e)}") 
                if attempt == max_retries - 1:
                    return f'{{"error": "Translation Failed - {str(e)}"}}'
                time.sleep(3) 
                    
        return '{"error": "Max retries reached."}'

# --- EXAMPLE USAGE FOR BACKEND INTEGRATION ---
if __name__ == "__main__":
    
   
    SHARED_API_KEY = "API" 
    
    # Initialize the reverse engine
    reverse_nlp = GlossToArabicEngine(api_key=SHARED_API_KEY)
    
    # Testing the noisy inputs!
    test_sentences = [
       "انا روح شراء تفاح صيدلية"  
    ]
    
    for test_gloss in test_sentences:
        raw_result = reverse_nlp.translate_to_arabic(test_gloss)
        
        print(f"Vision Gloss: {test_gloss}")
        
        try:
            # Parse the JSON string returned by the API
            result_json = json.loads(raw_result)
            print(f"Primary Output: {result_json.get('primary', '')}")
            print(f"Alternative Output: {result_json.get('alternative', '')}\n")
        except json.JSONDecodeError:
            print(f"Raw Output (Failed to parse JSON): {raw_result}\n")