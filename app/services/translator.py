import asyncio
import json
import asyncio
import logging
import functools

logger = logging.getLogger("babel.translator")

class ProviderUnavailableError(Exception):
    pass

def is_usable_translation(text) -> bool:
    if text is None:
        return False
    val = str(text).strip()
    if val == "":
        return False
    if val == "<i></i>":
        return False
    return True

def with_retry(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retries = 3
        backoffs = [5, 15, 30]
        for attempt in range(retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                recoverable = any(x in err_str for x in ["429", "500", "502", "503", "504", "timeout", "connection", "rate limit", "quota"])
                if not recoverable or attempt == retries:
                    if recoverable:
                        raise ProviderUnavailableError(f"Provider unavailable after {retries} retries: {str(e)}")
                    raise e

                wait_time = backoffs[attempt]
                logger.warning(f"Transient provider error in {func.__name__} (Attempt {attempt+1}/{retries}): {e}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
    return wrapper

import logging
import re
from typing import List, Optional
from google import genai
from google.genai import types
import openai
import httpx
import srt

from app.core.db import DB_PATH, get_setting, update_job, append_job_log, save_translation_memory, get_translation_memory

logger = logging.getLogger("babel.translator")

def get_system_instruction(target_language: str, glossary: str = "", show_title: str = "") -> str:
    glossary_section = ""
    if glossary and glossary.strip():
        glossary_section = "\n\nGLOSSARY - Always use these exact translations:\n" + glossary.strip() + "\n"

    show_context = ""
    if show_title:
        show_context = f"\nYou are translating subtitles for: \"{show_title}\". Adapt tone and terminology accordingly.\n"

    return f"""You are a professional film/TV subtitle translator translating from English to {target_language}.{show_context}
Translate the numbered subtitle blocks to natural, idiomatic {target_language}.
{glossary_section}
STRICT RULES:
1. Translate accurately and idiomatically into natural {target_language}.
2. Preserve all subtitle formatting tags exactly as they appear (e.g. <i>, </i>, and ASS tags like {{\\an8}}). Do not encode them into HTML entities.
3. If a block is empty or contains only '<i></i>', keep it exactly as '<i></i>'.
4. If a line starts with a speaker name or label in capital letters (e.g. ALICE:, OFFICER:), keep character names intact and only translate descriptive titles if appropriate, maintaining the colon separator.
5. You MUST return a JSON object with a key "translations" containing the array of objects with integer "id" and string "text".
6. Keep translations concise. Split lines naturally using "\n" if a line exceeds 42 characters, but NEVER exceed 2 lines per subtitle block. Combine or condense text if necessary.
Example:
{{"translations": [{{"id": 1, "text": "Translated text"}}, {{"id": 2, "text": "Translated text"}}]}}
"""

def extract_json_safely(raw_text: str) -> List[dict]:
    """Robust JSON parser that extracts translations even if formatting has minor hiccups."""
    raw_text = raw_text.strip()
    # 1. Direct parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and "translations" in data:
            return data["translations"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Repair common JSON issues before regex fallback
    try:
        repaired_text = re.sub(r'[\x00-\x1F]+', '', raw_text)
        repaired_text = re.sub(r',\s*([\]}])', r'\1', repaired_text)
        data = json.loads(repaired_text)
        if isinstance(data, dict) and "translations" in data:
            return data["translations"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. Regex match for "translations": [...]
    match = re.search(r'"translations"\s*:\s*(\[[\s\S]*?\])\s*\}?', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Fallback: individual regex search
    items = []
    pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
    for m in pattern.finditer(raw_text):
        try:
            item_id = int(m.group(1))
            item_text = m.group(2)
            items.append({"id": item_id, "text": item_text})
        except Exception:
            pass

    if items:
        return items

    raise ValueError(f"Could not extract JSON translation array from response: {raw_text[:100]}...")


def get_provider_capabilities(provider: str) -> dict:
    if provider == "deepl":
        return {
            "supports_structured_output": False,
            "supports_identical_classification": False,
            "supports_context": False
        }
    return {
        "supports_structured_output": True,
        "supports_identical_classification": True,
        "supports_context": True
    }

def validate_classifier_output(raw_text: str, items: list) -> list:
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Classifier validation: received {len(items)} candidates. Raw type: {type(raw_text)}")

    valid_results = []
    returned_ids = set()

    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        clean_text = "\n".join(lines).strip()

    try:
        data = json.loads(clean_text)

        # Handle dict or list root
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []
            logger.warning(f"Classifier validation: Unexpected JSON root type {type(data)}")

        logger.info(f"Classifier validation: parsed {len(results)} results from JSON")

        expected_ids = {item["id"]: item["text"] for item in items}
        allowed_reasons = {"proper_noun", "brand", "acronym", "number", "symbol", "non_verbal"}

        kept = 0
        translated = 0
        rejected = 0

        for r in results:
            if not isinstance(r, dict):
                logger.warning(f"Classifier validation: rejected result item of type {type(r)}")
                rejected += 1
                continue

            rid = r.get("id")
            act = str(r.get("action", "")).lower()
            if rid not in expected_ids:
                logger.warning(f"Classifier validation: rejected result for unknown ID {rid}")
                rejected += 1
                continue

            original_text = expected_ids[rid]

            if act == "keep":
                reason = str(r.get("reason", "")).lower()

                # Sanity checks
                is_valid_keep = True
                if reason not in allowed_reasons:
                    logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (invalid reason: {reason})")
                    is_valid_keep = False
                elif reason == "number":
                    if not any(c.isdigit() for c in original_text):
                        logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (reason=number but no digits)")
                        is_valid_keep = False
                elif reason in ["proper_noun", "brand"]:
                    # Deterministic plausibility validation (fail-closed)
                    import re
                    words = [w for w in re.split(r"[^a-zA-Z0-9]+", original_text) if w]

                    if len(words) > 4:
                        logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (reason={reason} but text is {len(words)} words)")
                        is_valid_keep = False
                    else:
                        minor_words = {"and", "of", "the", "in", "de", "la", "von", "van", "a", "an"}
                        for w in words:
                            if w.lower() in minor_words:
                                continue
                            # If a word is not purely digits and has no uppercase letters, it's not a valid proper noun
                            if not any(c.isupper() for c in w) and not w.isdigit():
                                logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (reason={reason} but word '{w}' is not capitalized)")
                                is_valid_keep = False
                                break
                elif reason == "non_verbal":
                    import re
                    has_bracket = bool(re.search(r"^[\[\(].*?[\]\)]$", original_text.strip()))
                    has_music = bool(re.search(r"[♪♬]", original_text))
                    words = [w for w in re.split(r"[^a-zA-Z0-9]+", original_text) if w and any(c.isalpha() for c in w)]
                    is_purely_symbolic = len(words) == 0

                    if not (has_bracket or has_music or is_purely_symbolic):
                        logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (reason={reason} but lacks determinable format like brackets)")
                        is_valid_keep = False
                elif reason == "acronym":
                    import re
                    words = [w for w in re.split(r"[^a-zA-Z0-9]+", original_text) if w and any(c.isalpha() for c in w)]
                    if not words:
                        pass
                    elif any(not any(c.isupper() for c in w) for w in words):
                        logger.info(f"Classifier validation: ID {rid} downgraded KEEP->TRANSLATE (reason={reason} but words are not uppercase)")
                        is_valid_keep = False

                if not is_valid_keep:
                    act = "translate"

            if act == "keep": kept += 1
            elif act == "translate":
                translated += 1
                # Enforce that translate text is actually provided and doesn't just echo source
                # We do this by checking if the text matches original. If so, we clear it so fallback logic triggers.
                provided_text = str(r.get("text", "")).strip()
                if provided_text and provided_text == original_text.strip():
                    logger.info(f"Classifier validation: ID {rid} TRANSLATE echoes source, clearing text to force recovery")
                    r["text"] = ""
            else:
                logger.warning(f"Classifier validation: rejected result for ID {rid} due to invalid action {act}")
                rejected += 1
                continue

            valid_results.append({"id": rid, "action": act, "reason": r.get("reason", ""), "text": r.get("text", "")})
            returned_ids.add(rid)

        logger.info(f"Classifier validation: Validated {kept} KEEP, {translated} TRANSLATE, {rejected} REJECTED")
    except Exception as e:
        logger.error(f"Classifier validation: JSON parse failed: {e}")

    # Failsafe: Any missing items must be translated
    for item in items:
        if item["id"] not in returned_ids:
            logger.info(f"Classifier validation: Failsafe triggered for ID {item['id']}, forcing TRANSLATE")
            valid_results.append({"id": item["id"], "action": "translate", "reason": "malformed_fallback", "text": item["text"]})

    return valid_results

class SubtitleTranslator:
    @with_retry
    async def classify_and_recover_identical(self, items: list, target_language: str, show_title: str) -> list:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "deepl":
            return [] # fallback to translation

        system_prompt = f"""You are a subtitle quality assurance AI for {target_language}.
The following lines were identical in English and {target_language}.
Decide for each line whether it should be KEPT identical (e.g. proper nouns, brands, numbers, untranslatable sounds) or TRANSLATED.
If a line is a song lyric, classify it as TRANSLATE. Do not keep song lyrics in English unless it is an untranslatable proper noun.

KEEP -> text may remain identical
TRANSLATE -> text MUST contain the actual translated target-language text, never merely the source.

Return ONLY a JSON object with a single key 'results' containing an array of objects.
Valid reasons for KEEP: proper_noun, brand, acronym, number, symbol, non_verbal.
"""
        prompt = f"Context: {show_title}\n\nLines:\n" + json.dumps(items, ensure_ascii=False)

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "action": {"type": "STRING", "enum": ["keep", "translate"]},
                            "reason": {"type": "STRING", "enum": ["proper_noun", "brand", "acronym", "number", "symbol", "non_verbal", "none"]},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "action", "text"]
                    }
                }
            },
            "required": ["results"]
        }

        # We will reuse the client calls directly here to keep it self-contained
        if provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = get_setting("gemini_api_key", "")
            model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
            client = genai.Client(api_key=api_key)

            def do_gemini():
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, do_gemini)
            return validate_classifier_output(resp.text, items)

        elif provider == "openai":
            import openai
            api_key = get_setting("openai_api_key", "")
            model_name = get_setting("openai_model", "gpt-4o-mini")
            client = openai.Client(api_key=api_key)

            def do_openai():
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, do_openai)
            try:
                content = resp.choices[0].message.content
                return validate_classifier_output(content, items)
            except Exception:
                return []

        elif provider == "ollama":
            import httpx
            ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
            model_name = get_setting("ollama_model", "llama3")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name, "prompt": full_prompt, "format": "json", "stream": False}
                )
                try:
                    return validate_classifier_output(resp.json()["response"], items)
                except Exception:
                    return []

        return []
    def get_gemini_client(self):
        api_key = get_setting("gemini_api_key", "")
        if not api_key:
            raise ValueError("Gemini API Key is not configured in settings.")
        return genai.Client(api_key=api_key)

    def get_openai_client(self):
        api_key = get_setting("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API Key is not configured in settings.")
        return openai.OpenAI(api_key=api_key)

    @with_retry
    async def escalate_single_line(self, target_idx: int, target_text: str, prev_text: str, next_text: str, target_language: str, show_title: str) -> str:
        import logging
        logger = logging.getLogger(__name__)

        provider = get_setting("ai_provider", "gemini").lower()

        escalate_enabled = get_setting("escalate_to_pro", "false").lower() == "true"
        esc_provider = get_setting("escalation_provider", "none").lower()
        if escalate_enabled and esc_provider != "none":
            provider = esc_provider

        if provider == "deepl":
            return target_text

        system_prompt = f"You are a subtitle translator. Translate the TARGET line to {target_language}. The Previous and Next lines are for context only. Return a JSON object with a single key 'translation' containing the translated string."
        prompt = f"Context: {show_title}\n\nPrevious: {prev_text}\nTARGET: {target_text}\nNext: {next_text}\n\nTranslate TARGET:"

        schema = {
            "type": "OBJECT",
            "properties": {
                "translation": {"type": "STRING"}
            },
            "required": ["translation"]
        }

        def _safe_parse(raw_resp: str) -> str:
            clean_text = raw_resp.strip()
            if clean_text.startswith("```"):
                lines = clean_text.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                clean_text = "\n".join(lines).strip()
            try:
                data = json.loads(clean_text)
                res = data.get("translation", "").strip()
                if res and res != target_text:
                    logger.info(f"Escalation line {target_idx}: valid translation returned ({len(res)} chars)")
                    return res
                elif res == target_text:
                    logger.info(f"Escalation line {target_idx}: returned identical text")
                    return target_text
                else:
                    logger.info(f"Escalation line {target_idx}: returned empty translation")
                    return target_text
            except Exception as e:
                logger.error(f"Escalation line {target_idx} JSON parse failed: {e}. Raw: {raw_resp[:50]}")
                return target_text

        try:
            if provider == "gemini":
                from google import genai
                from google.genai import types
                import asyncio
                api_key = get_setting("gemini_api_key", "")

                model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
                if escalate_enabled and esc_provider == "gemini":
                    esc_model = get_setting("escalation_model", "")
                    if esc_model:
                        model_name = esc_model

                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=schema,
                )
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.models.generate_content(model=model_name, contents=prompt, config=config))
                return _safe_parse(resp.text)

            elif provider == "openai":
                import openai
                import asyncio
                api_key = get_setting("openai_api_key", "")

                model = get_setting("openai_model", "gpt-4o-mini")
                if escalate_enabled and esc_provider == "openai":
                    esc_model = get_setting("escalation_model", "")
                    if esc_model:
                        model = esc_model

                client = openai.OpenAI(api_key=api_key)
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_schema", "json_schema": {"name": "esc", "schema": schema, "strict": True}}
                ))
                return _safe_parse(resp.choices[0].message.content)
            elif provider == "deepl":
                import httpx
                api_key = get_setting("deepl_api_key", "")
                from app.core.languages import get_language
                lang_obj = get_language(lang_name)
                target_lang_code = lang_obj.deepl_code if lang_obj else lang_name.upper()[:2]
                url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    resp = await http_client.post(
                        url,
                        headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                        json={"text": [target_text], "target_lang": target_lang_code, "source_lang": "EN"}
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    res = data["translations"][0]["text"].strip()
                    if res and res != target_text:
                        return res
                    return target_text
        except Exception as e:
            logger.error(f"Escalation line {target_idx} API call failed: {e}")
            raise ProviderUnavailableError(f"Escalation failed: {e}") from e

        return target_text

    @with_retry
    async def translate_batch_gemini(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "") -> List[dict]:
        client = self.get_gemini_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'

        prompt = f"Translate the following {len(items)} subtitle lines into {target_language}:{context_section}\n\n" + json.dumps(items, ensure_ascii=False)

        schema = {
            "type": "OBJECT",
            "properties": {
                "translations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["id", "text"]
                    }
                }
            },
            "required": ["translations"]
        }

        glossary = get_setting("glossary", "")
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(target_language, glossary=glossary, show_title=show_title),
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        )

        def call_gemini(model_to_use):
            return client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=config
            )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: call_gemini(model_name))
        except Exception as e:
            # Automatic Fallback if Google deprecated or changed model name
            if "404" in str(e) or "not found" in str(e).lower():
                fallback_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
                for fb in fallback_models:
                    if fb != model_name:
                        try:
                            logger.warning(f"Model {model_name} failed with 404, falling back to {fb}")
                            response = await loop.run_in_executor(None, lambda: call_gemini(fb))
                            break
                        except Exception:
                            continue
                else:
                    raise e
            else:
                raise e

        return extract_json_safely(response.text)

    @with_retry
    async def translate_batch_openai(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "") -> List[dict]:
        client = self.get_openai_client()

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'

        prompt = f"Translate the following {len(items)} subtitle lines into {target_language}:{context_section}\n\n" + json.dumps(items, ensure_ascii=False)

        glossary = get_setting("glossary", "")

        def call_openai(model_to_use):
            return client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": get_system_instruction(target_language, glossary=glossary, show_title=show_title)},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: call_openai(model_name))
        except Exception as e:
            if "404" in str(e) or "model_not_found" in str(e).lower():
                logger.warning(f"Model {model_name} failed with 404, falling back to gpt-4o-mini")
                response = await loop.run_in_executor(None, lambda: call_openai("gpt-4o-mini"))
            else:
                raise e

        return extract_json_safely(response.choices[0].message.content)

    @with_retry
    async def translate_batch_deepl(self, items: List[dict], target_language: str, context_lines: List[dict] = None) -> List[dict]:
        api_key = get_setting("deepl_api_key", "")
        if not api_key:
            raise ValueError("DeepL API Key is not configured.")

        from app.core.languages import get_language
        lang_obj = get_language(target_language)
        target_lang_code = lang_obj.deepl_code if lang_obj else target_language.upper()[:2]
        url = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"

        texts = [it["text"] for it in items]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                json={"text": texts, "target_lang": target_lang_code, "source_lang": "EN"}
            )
            resp.raise_for_status()
            data = resp.json()
            translations = data.get("translations", [])
            if len(translations) != len(items):
                raise ValueError(f"DeepL returned {len(translations)} translations, but expected {len(items)}.")
            return [{"id": items[i]["id"], "text": translations[i]["text"]} for i in range(len(items))]

    @with_retry
    async def translate_batch_ollama(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None, show_title: str = "") -> List[dict]:
        ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")

        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'

        glossary = get_setting("glossary", "")

        prompt = f"{get_system_instruction(target_language, glossary=glossary, show_title=show_title)}\n\nTranslate the following JSON list into {target_language}:{context_section}\n{json.dumps(items, ensure_ascii=False)}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name or "llama3", "prompt": prompt, "format": "json", "stream": False}
            )
            resp.raise_for_status()
            data = resp.json()
            return extract_json_safely(data.get("response", "{}"))

    async def translate_batch(self, items: List[dict], target_language: str = "English", context_lines: List[dict] = None, show_title: str = "") -> List[dict]:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "openai":
            model = get_setting("openai_model", "gpt-4o-mini")
            return await self.translate_batch_openai(items, target_language, model, context_lines=context_lines, show_title=show_title)
        elif provider == "deepl":
            return await self.translate_batch_deepl(items, target_language, context_lines=context_lines)
        elif provider in ["ollama", "localai"]:
            model = get_setting("ollama_model", "llama3")
            return await self.translate_batch_ollama(items, target_language, model, context_lines=context_lines, show_title=show_title)
        else:
            model = get_setting("gemini_model", "gemini-3.5-flash-lite")
            return await self.translate_batch_gemini(items, target_language, model, context_lines=context_lines, show_title=show_title)

    async def translate_srt_content(
        self,
        subs: List[srt.Subtitle],
        target_language: str = "English",
        batch_size: int = 50,
        job_id: Optional[int] = None,
        show_title: Optional[str] = None
    ) -> List[srt.Subtitle]:
        total_lines = len(subs)
        batches = []
        for i in range(0, total_lines, batch_size):
            chunk = subs[i:i + batch_size]
            batch_payload = [{"id": j, "text": sub.content} for j, sub in enumerate(chunk, start=i)]
            batches.append((i, chunk, batch_payload))

        translated_subs = [
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=sub.content)
            for sub in subs
        ]

        import os
        from app.core.languages import get_language
        data_dir = os.path.dirname(DB_PATH)
        lang_obj = get_language(target_language)
        lang_code = lang_obj.code if lang_obj else target_language.lower()[:2]
        partial_file = os.path.join(data_dir, f"job_{job_id}_{lang_code}_partial.json") if job_id else None

        import hashlib
        fingerprint = hashlib.md5("".join(s.content for s in subs).encode("utf-8")).hexdigest() + "_" + lang_code

        partial_dict = {}
        if partial_file and os.path.exists(partial_file):
            try:
                with open(partial_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("fingerprint") == fingerprint:
                    lines_data = data.get("lines", {})
                    partial_dict = {int(k): v for k, v in lines_data.items()}
                    for k, v in partial_dict.items():
                        if k < len(translated_subs):
                            translated_subs[k].content = v
                    logger.info(f"Loaded partial progress for job {job_id} ({len(partial_dict)} lines)")
                else:
                    logger.warning(f"Partial progress fingerprint mismatch for job {job_id}. Discarding.")
            except Exception as e:
                logger.error(f"Failed to load partial progress for job {job_id}: {e}")

        processed_count = 0
        context_window_size = 5
        previous_context = []
        if show_title:
            try:
                tm_context = get_translation_memory(show_title, limit=10)
                if tm_context:
                    previous_context.extend(tm_context)
            except Exception:
                pass

        for batch_idx, (start_idx, chunk, payload) in enumerate(batches):
            end_idx = min(start_idx + len(payload), total_lines)

            # --- ÄKTA RESUME: Skip batch if all lines are already translated ---
            if payload and all(p["id"] in partial_dict for p in payload):
                processed_count += len(payload)
                if job_id:
                    update_job(job_id, processed_lines=processed_count, current_batch=f"Skipping cached lines {start_idx + 1}-{end_idx} / {total_lines}")
                continue
            # -------------------------------------------------------------------

            all_empty = all(p["text"].strip() == "<i></i>" for p in payload)
            if all_empty:
                processed_count += len(payload)
                if job_id:
                    update_job(job_id, processed_lines=processed_count, current_batch=f"Lines {start_idx + 1}-{end_idx} / {total_lines}")
                continue

            if job_id:
                update_job(job_id, current_batch=f"Translating lines {start_idx + 1}-{end_idx} of {total_lines}")

            try:
                # --- Send only the missing cues to provider ---
                missing_payload = [p for p in payload if p["id"] not in partial_dict]
                res_dict = {}
                if missing_payload:
                    results = await self.translate_batch(
                        missing_payload,
                        target_language=target_language,
                        context_lines=previous_context if previous_context else None,
                        show_title=show_title or ""
                    )
                    res_dict = {r["id"]: r["text"] for r in results if "id" in r and "text" in r and is_usable_translation(r["text"])}
                
                # Merge back the already solved cues
                for p in payload:
                    if p["id"] in partial_dict:
                        res_dict[p["id"]] = partial_dict[p["id"]]

                # --- BABEL SMART RECOVERY (QA ENGINE) ---
                missing_ids = [p["id"] for p in payload if p["id"] not in res_dict]
                if missing_ids:
                    logger.warning(f"QA: AI dropped {len(missing_ids)} lines in batch. Triggering Smart Recovery.")
                    if job_id:
                        append_job_log(job_id, f"QA Alert: AI skipped {len(missing_ids)} lines. Triggering Smart Recovery for IDs: {missing_ids[:5]}...")

                    missing_payload = [p for p in payload if p["id"] in missing_ids]

                    # Micro-request for ONLY the missing lines
                    try:
                        recovery_results = await self.translate_batch(
                            missing_payload,
                            target_language=target_language,
                            context_lines=previous_context if previous_context else None,
                            show_title=show_title or ""
                        )
                        for r in recovery_results:
                            if "id" in r and "text" in r and is_usable_translation(r["text"]):
                                res_dict[r["id"]] = r["text"]

                        still_missing = [p["id"] for p in payload if p["id"] not in res_dict]
                        if not still_missing and job_id:
                            append_job_log(job_id, f"QA Success: Smart Recovery successfully translated the missing lines!")
                    except Exception as e:
                        logger.error(f"Smart recovery failed: {e}")
                        if job_id:
                            append_job_log(job_id, f"QA Warning: Smart Recovery failed ({e}). Proceeding anyway.")
                # ----------------------------------------

                for p in payload:
                    idx = p["id"]
                    if p["text"].strip() == "<i></i>":
                        translated_subs[idx].content = "<i></i>"
                        partial_dict[idx] = "<i></i>"
                    elif idx in res_dict:
                        translated_subs[idx].content = res_dict[idx]
                        partial_dict[idx] = translated_subs[idx].content

                if partial_file:
                    try:
                        wrapper = {"fingerprint": fingerprint, "lines": partial_dict}
                        tmp_file = partial_file + ".tmp"
                        with open(tmp_file, "w", encoding="utf-8") as f:
                            json.dump(wrapper, f, ensure_ascii=False)
                        os.replace(tmp_file, partial_file)
                    except Exception as e:
                        logger.error(f"Failed to save partial progress for job {job_id}: {e}")

                valid_translations = [
                    {"original": p["text"], "translated": res_dict[p["id"]]}
                    for p in payload if p["id"] in res_dict and p["text"].strip() != "<i></i>"
                ]
                if valid_translations:
                    previous_context = valid_translations[-context_window_size:]

                processed_count += len(payload)
                if job_id:
                    update_job(job_id, processed_lines=processed_count)

            except ProviderUnavailableError as e:
                # Let pipeline handle WAITING_PROVIDER
                if job_id:
                    update_job(job_id, processed_lines=processed_count)
                raise e
            except Exception as e:
                logger.error(f"Batch {start_idx} failed: {e}")
                processed_count += len(payload)
                if job_id:
                    append_job_log(job_id, f"Warning: Lines {start_idx + 1}-{end_idx} could not be translated: {e}. Keeping original text.")
                    update_job(job_id, processed_lines=processed_count)

        return translated_subs

