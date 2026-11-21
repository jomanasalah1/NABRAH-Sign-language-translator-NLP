import time
import requests

class ArabicNLPEngine:
    def __init__(self, api_key: str = "API"):
        """
        Initializes the translation engine with a closed-vocabulary dictionary lock
        derived exactly from the project's complete video dataset annotations.
        """
        # Using the flash-lite model to safely avoid free-tier rate limits
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
        
        # =====================================================================
        # 1. EXACT DATASET VOCABULARY LOCK (Fully Loaded)
        # =====================================================================
        self.dataset_words = [
            "قلق", "قلم", "قطر", "عدوى", "العجمي", "عايز", "عامل ايه", "العامرية", 
            "عقد", "عقد عمل", "عشرة", "عوايد", "عين", "عنب", "عشرين", "علوم", 
            "خمسة", "خمسين", "خبرة", "خروج", "حق", "حامل", "حاسب آلي", "غالي", 
            "غياب", "أفريقيا", "أكل", "العصافرة", "الإبراهيمية", "المنشية", "بحث", 
            "بيض", "كليوباترا", "إجابة", "الترام", "إنجاز", "فيسبوك", "فضة", 
            "فصل", "فرنسا", "جليم", "إنستجرام", "كبد", "كيمياء", "معاش", "ملف طبي", 
            "مهندس", "مكافأة", "ممرض", "نقاش", "نجار", "نتيجة التحاليل", "جواز سفر", 
            "واتساب", "عباية", "أبيض", "أبو تلات", "أبو قير", "قديم", "عيشة", 
            "أجازة", "أحمر", "قهوة", "عيش", "أقدام", "ألف", "عملية", "قميص", 
            "أنف", "عربية", "عربي", "أربعة", "أشياء", "أسمر", "أسنان", "عصر", 
            "عصير", "أزرق", "بكتيريا", "بدلة", "بحري", "بلح", "بلاط", "بنزين", 
            "بشرة بيضاء", "بصل", "بطاطس", "بطن", "بكم", "تربية نوعية", "بطاقة", 
            "فم", "بكرة", "برتقال", "بيطاش", "سينما الإبراهيمية", "دخول", "ضغط شغل", 
            "ضغط", "دقيقة", "درجات", "ضرب", "درس", "دوام", "ضايع", "دين", 
            "ظهر", "دوام", "إعدادي", "إبرة", "ابتدائي", "ايه", "العمر", "الخميس", 
            "الأربعاء", "الاتنين", "الجمعة", "الأحد", "النهاردة", "العمر", "السبت", 
            "الشاطبي", "التلات", "التزام", "الطول", "الوزن", "امشي", "امتحان", 
            "إنجليزي", "إنتاج", "انتظار", "إسعاف", "إشارة مرور", "اشتري", "اسم المريض", 
            "إسرائيل", "أسود", "اتنين", "فجر", "فهم", "فاكهة", "فريق", "فيروس", 
            "فين", "فلوس", "فستان", "في", "جدول", "جمع", "جراج", "كرافتة", 
            "جزر", "جزمة", "جزمة فرح", "جديد", "جبنة", "جيبة", "جيش", "جهاز تنفس", 
            "جلد", "غبي", "غياب", "غرفة العمليات", "جغرافيا", "جوا", "جي", 
            "حق", "هدف", "حادثة", "حامل", "حرارة", "حساسية", "حكومة", "حساب", 
            "حزام", "هدوم صيفي", "هدوم شتوي", "حضور", "علاج", "إنترنت", "جاكيت", 
            "جامعة", "جينز", "جهاز تنفس", "جرح", "كعب", "كذب", "كشف", "كسر", 
            "كتاب", "خطر", "خبرة", "خيار", "خضار", "خروج", "كنيسة", "كوبري", 
            "كلية", "كراسة", "كتير", "لا", "لبن", "لحمة", "لون", "معمل", 
            "مع", "معلومة", "المعمورة", "مدح", "مدرسة", "مجهود", "المغرب", "محل", 
            "محطة", "محطة الرمل", "مكرونة", "مقص", "مخنوق", "مكتب", "مكتبة", 
            "ملابس", "ملح", "ممنوع", "المندرة", "مريض", "مساء", "مساء الخير", 
            "مشروع", "مسموح", "مصنع", "مطافي", "موقف", "منور", "مرور", "متأخر", 
            "متين", "مياه", "ميامي", "ميكروباص", "مقابلة شغل", "مؤهل دراسي", "معلم", 
            "محاضرة", "محاسب", "مدرس", "مدير", "محرم بك", "المنتزه", "موز", 
            "مقابلة", "مراجعة", "مرتب", "مستقبل", "متابعة", "موتوسيكل", "مسيحي", 
            "مش نظيف", "متعصب", "متمزج", "ممرضة", "مسلم", "مستشفى", "نظارة", 
            "نافع", "نجاح", "ناجح", "ناموس", "نشاط", "نظيف", "نسيان", "نظام", 
            "قدام", "قليل", "قصير", "أوتوبيس", "بيجامة", "قلي", "قانون", "راسب", 
            "رصيف", "رسم", "رخيص", "رضاعة", "رجل", "روح", "رفيع", "رشدي", 
            "رز", "روشتة", "رياضيات", "ساقع", "سائق", "صعب", "ساعة", "سبعة", 
            "صباح", "صباح الخير", "سبورة", "سفر", "سهل", "صيدلية", "سلطة", 
            "سمك", "سنة", "ثانوي", "صيانة", "سرير", "ستة", "شعر", "شحن", 
            "شهادة", "شهر", "شنطة", "شرح", "شرح أونلاين", "شارع", "شطه", 
            "شاي", "شبشب", "شمال", "شركة", "شركة خاصة", "شغل", "شكرا", 
            "شرب", "شوربة", "شرطة", "شراب", "سيدي جابر", "سموحة", "سؤال", 
            "صداع", "سكر", "سخن", "سرعة", "سوق", "تأخير", "تأمين", "تأمين صحي", 
            "تعاون", "تعب", "تعليم", "تخدير", "طبيب", "تدريب", "تاجر", "تحليل", 
            "طالب", "طنطا", "ترقية", "طرح", "تربية", "تاريخ", "ترجمة", "طرحة", 
            "تركيز", "تسجيل", "تشخيص", "تطعيم", "تطبيق", "طوارئ", "طويل", "توظيف", 
            "تاكسي", "تذكرة", "تجارة", "تخين", "تسعة", "ثلاثة", "ثمانية", "طماطم", 
            "تفاح", "ثوم", "تيشيرت", "فيكتوريا", "فيروس", "وقت", "واجب", "واحيد", 
            "وقت", "وراء", "الورديان", "واسع", "ودن", "ودن كبيرة", "وصول", "يمشي", 
            "يمين", "يوم", "زعلان", "زحمة", "زميل", "زميل شغل", "زبدة", "زيت", 
            "ذراع", "زراعة", "حجز موعد", "طاقية", "مصطفى كامل", 
            "كيلو 21", "كل سنة و أنت طيب", "مكتبة الإسكندرية", 
            "مناخير كبيرة", "متعصب جدا", "رضاعة صناعي", 
            "سان ستيفانو", "سبوع", "سيدي بشر", "شهادة ميلاد", "شهادة تجنيد", 
            "شرح بلغة الإشارة", "تربية نوعية", "يقشعر"
        ]
        
        # Clean up any accidental duplicates from the raw list and join them
        self.dataset_words = list(dict.fromkeys(self.dataset_words))
        self.vocab_string = "، ".join(self.dataset_words)


      # =====================================================================
        # 2. SYSTEM PROMPT DESIGN & MASTER RULES
        # =====================================================================
        self.system_prompt = f"""You are a strict rule-based translator that converts Egyptian Arabic sentences into Egyptian Sign Language (EgySL) gloss tokens.

═══════════════════════════════════════════════════════
SECTION 1 — ABSOLUTE FORBIDDEN ACTIONS (never break these)
═══════════════════════════════════════════════════════
❌ NEVER add any word that does not exist in the input sentence.
❌ NEVER add explanation, punctuation, periods, commas, or question marks.
❌ NEVER write any English words.
❌ NEVER repeat a pronoun (أنا/هو/هي/إحنا) more than once per clause.
❌ NEVER decompose family words — خال، خالة، عم، عمة، جد، ست stay exactly as they are.
❌ NEVER invent synonyms or paraphrase — if a word must stay, keep it exactly.
❌ NEVER DELETE CONTENT WORDS: You must keep ALL main verbs, nouns, and adjectives from the input sentence. Do NOT drop a word just because it is missing from the Priority Vocabulary List.

═══════════════════════════════════════════════════════
SECTION 2 — TRANSFORMATION RULES (apply in this exact order)
═══════════════════════════════════════════════════════

STEP 1 — VOCABULARY SUBSTITUTION (do this BEFORE anything else):
   For every content word in the input, follow this decision tree:
   → Is the word exactly in the Priority Vocabulary List? → USE the list word.
   → Is there a word in the Priority Vocabulary List with the SAME or VERY SIMILAR meaning? → REPLACE it with that list word.
   → If the word is NOT in the list → YOU MUST KEEP IT. Convert it to its base root (المصدر) and include it in the final output. Do NOT delete it.

STEP 2 — DELETE these word types (they have no sign):
   • All prepositions: من، إلى، على، عن، ل، ب (Do NOT delete main words starting with ب like بروح, keep the root روح)
   • All conjunctions and connectors: و، أو، لكن، عشان، علشان، لأن، إن، أن، اللي، اللي بيـ
   • Verb to-be: كان، يكون، بيكون، هيكون
   • Definite article: ال (remove from the beginning of each word)
   • Tense markers only when the tense is clear from context: هـ (future prefix), بيـ (present prefix)

STEP 3 — VERB FORM & ORIGIN ROOT (المصدر / الأصل): 
   Convert ALL verbs and complex words to their simplest base root, verbal noun (المصدر), or raw origin. This applies to EVERY word in the sentence, even if it is not in the vocabulary list!
   "اكتشف" / "يكتشف" / "سأكتشف" → "اكتشاف"
   "أنا رحت" / "بروح" / "هروح" / "سأذهب" → "روح"
   "إحنا اشترينا" / "بنشتري" → "شراء" أو "شرى"
   "استخدم" / "يستخدم" → "استخدام"

STEP 4 — NEGATION: Replace (مش / لن / لم / ما / ميـ) with كلمة "لا" وضعها في آخر الجملة.
   "أنا مش عارف" → "أنا عارف لا"

STEP 5 — WORD ORDER (SOV — Subject, Object, Verb):
   Rearrange remaining words: Subject first → Object/Complement second → Verb last.
   Time words and question words go at the END after the verb.

STEP 6 — QUESTION WORDS: Move (فين، مين، إيه، كام، إزاي، ليه، امتى) to the very END.

STEP 7 — POSSESSIVES: Owner comes first, then the noun.
   "اسمه" → "هو اسم"
   "بيتنا" → "إحنا بيت"

STEP 8 — GENDER: All output must be masculine. Never use feminine verb forms or suffixes.

═══════════════════════════════════════════════════════
SECTION 3 — EXAMPLES (study these carefully)
═══════════════════════════════════════════════════════

Input:  "أنا سأذهب إلى المستشفى عشان أشتري سيارة"
Output: "أنا مستشفى عربية شرى روح"
Why: إلى removed, عشان removed, سيارة→عربية (vocab list), سأذهب→روح

Input:  "هو مش عايز يروح المدرسة بكرة"
Output: "هو مدرسة بكرة روح عايز لا"
Why: مش→لا at end, يروح→روح, ال removed from المدرسة

Input:  "إنت رايح فين دلوقتي"
Output: "إنت رايح فين دلوقتي"
Why: question word stays at end, no verbs to root, no deletions needed

Input:  "خالي جه من إسكندرية"
Output: "خال إسكندرية جي"
Why: خال stays as-is (family word rule), ي possessive removed = just خال, من removed, جه→جي

Input:  "أنا و التلميذ روحنا نشرب قهوة"
Output: "أنا طالب قهوة شرب روح"
Why: و removed, التلميذ→طالب (similar meaning in vocab list), ال removed, روحنا→روح, نشرب→شرب

Input:  "عنده ثلاث كتب في الأوضة"
Output: "هو أوضة كتاب 3"
Why: عنده→هو (possessive rule), ثلاث→3, في removed, ال removed, plural كتب→كتاب

Input:  "أنا مش فاهم الكلام ده"
Output: "أنا كلام فاهم لا"
Why: مش→لا at end, ال removed, ده kept (demonstrative = content word)

Input:  "إحنا كنا في الجامعة"
Output: "إحنا جامعة"
Why: كنا (to-be) deleted, في deleted, ال deleted

Input:  "الراجل اللي اشتغل في الشركة"
Output: "راجل شركة شغل"
Why: اللي deleted, في deleted, ال deleted from both, اشتغل→شغل

Input:  "جده صاحي بدري"
Output: "جد صاحي بدري"
Why: جد stays as-is, possessive ه removed = جد, صاحي and بدري kept

═══════════════════════════════════════════════════════
SECTION 4 — PRIORITY VOCABULARY LIST
═══════════════════════════════════════════════════════
These are the ONLY words the avatar can show. Use them whenever their meaning matches:
[{self.vocab_string}]

═══════════════════════════════════════════════════════
SECTION 5 — FINAL OUTPUT RULE (most important)
═══════════════════════════════════════════════════════
Output ONLY the gloss words separated by single spaces.
Do NOT output a 'Why:' explanation in your final response.
No punctuation. No explanation. No extra words. No English.
If you are uncertain about any transformation, keep the original word unchanged.
REMEMBER: Every output word must have come from the input — you cannot invent words.
"""

    def translate(self, arabic_text: str, max_retries: int = 4) -> str:
        if not arabic_text or not arabic_text.strip():
            return ""

        payload = {
            "system_instruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"parts": [{"text": arabic_text.strip()}]}],
            "generationConfig": {
                "temperature": 0.1,  
                "maxOutputTokens": 1024
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload, 
                    timeout=30 # 1. INCREASED TIMEOUT: Gives the large prompt time to process
                )
                
                if response.status_code == 429:
                    # Let's read exactly what Google's excuse is
                    print(f" [429 Error Details]: {response.text}")
                    time.sleep((2 ** attempt) + 3)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

            except Exception as e:
                # 2. PRINT ERROR: Let's see exactly what Google is complaining about
                print(f" Attempt {attempt + 1} Failed: {str(e)}") 
                
                if attempt == max_retries - 1:
                    return f"ERROR: Translation Failed - {str(e)}"
                
                # 3. ADDED SLEEP: Stops the code from burning 4 retries in 0.1 seconds
                time.sleep(3) 
                    
        return "ERROR: Max retries reached."

# --- FOR LOCAL VERIFICATION ---
if __name__ == "__main__":
    nlp = ArabicNLPEngine()
    
    test_sentence = "إنت حطيت قلمي فين على المكتب؟"
    result = nlp.translate(test_sentence)
    
    # 1. Try to print it to the terminal
    print(f"\nTerminal Output: {result}")
    
    # 2. Save it to a text file to bypass terminal bugs
    with open("translation_result.txt", "w", encoding="utf-8") as file:
        file.write(result)
        
    print("\n Saved to translation_result.txt. Open that file to check the true output!")