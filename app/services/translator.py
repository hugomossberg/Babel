import asyncio
import json
import logging
import re
from typing import List, Optional
from google import genai
from google.genai import types
import openai
import httpx
import srt

from app.core.db import get_setting, update_job, append_job_log

logger = logging.getLogger("babel.translator")

def get_system_instruction(target_language: str = "Swedish", glossary: str = "", show_title: str = "") -> str:
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
6. Keep translations concise. Split lines naturally if a single line exceeds 42 characters.
Example:
{{"translations": [{{"id": 1, "text": "Hej"}}, {{"id": 2, "text": "Världen"}}]}}
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

class SubtitleTranslator:
    async def classify_and_recover_identical(self, items: list, target_language: str, show_title: str) -> list:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "deepl":
            return [] # fallback to translation
            
        system_prompt = f"""You are a subtitle quality assurance AI for {target_language}.
The following lines were identical in English and {target_language}.
Decide for each line whether it should be KEPT identical (e.g. proper nouns, brands, numbers, untranslatable sounds) or TRANSLATED.
If a line is a song lyric, classify it as TRANSLATE. Do not keep song lyrics in English unless it is an untranslatable proper noun.

Return ONLY a JSON array with this exact structure:
[
  {{
    "id": 123,
    "action": "keep",
    "reason": "proper_noun",
    "text": "Seth Cohen"
  }},
  {{
    "id": 124,
    "action": "translate",
    "text": "The translated text here"
  }}
]"""
        prompt = f"Context: {show_title}\n\nLines:\n" + json.dumps(items, ensure_ascii=False)
        
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
                        temperature=0.1
                    )
                )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, do_gemini)
            try:
                return json.loads(resp.text)
            except Exception:
                return []
                
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
                data = json.loads(content)
                if isinstance(data, dict):
                    # In case they wrapped it
                    for k, v in data.items():
                        if isinstance(v, list): return v
                return data
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
                    return json.loads(resp.json()["response"])
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

    async def escalate_single_line(self, target_idx: int, target_text: str, prev_text: str, next_text: str, target_language: str, show_title: str) -> str:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "deepl":
            return target_text

        system_prompt = f"You are a subtitle translator. Translate the TARGET line to {target_language}. The Previous and Next lines are for context only. Return ONLY the translated string for the TARGET line, no JSON, no quotes."
        prompt = f"Context: {show_title}\n\nPrevious: {prev_text}\nTARGET: {target_text}\nNext: {next_text}\n\nTranslate TARGET:"

        try:
            escalate_enabled = get_setting("escalate_to_pro", "false").lower() == "true"
            if provider == "gemini":
                from google import genai
                from google.genai import types
                import asyncio
                api_key = get_setting("gemini_api_key", "")
                model_name = get_setting("gemini_model", "gemini-3.5-flash-lite")
                if escalate_enabled and "lite" in model_name:
                    model_name = "gemini-3.6-flash"
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                )
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.models.generate_content(model=model_name, contents=prompt, config=config))
                return resp.text.strip()
            elif provider == "openai":
                import openai
                import asyncio
                api_key = get_setting("openai_api_key", "")
                model = get_setting("openai_model", "gpt-4o-mini")
                if escalate_enabled and "mini" in model:
                    model = "gpt-4o"
                client = openai.OpenAI(api_key=api_key)
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature=0.1))
                return resp.choices[0].message.content.strip()
        except Exception as e:
            return target_text
        return target_text

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

    async def translate_batch_deepl(self, items: List[dict], target_language: str, context_lines: List[dict] = None) -> List[dict]:
        api_key = get_setting("deepl_api_key", "")
        if not api_key:
            raise ValueError("DeepL API Key is not configured.")
        
        DEEPL_LANG_MAP = {
            "swedish": "SV", "danish": "DA", "norwegian": "NB", "finnish": "FI",
            "german": "DE", "french": "FR", "spanish": "ES", "italian": "IT",
            "dutch": "NL", "polish": "PL", "portuguese": "PT-PT", "russian": "RU",
            "japanese": "JA", "chinese": "ZH", "korean": "KO", "turkish": "TR",
            "czech": "CS", "romanian": "RO", "hungarian": "HU", "bulgarian": "BG",
            "greek": "EL", "indonesian": "ID", "arabic": "AR",
        }
        target_lang_code = DEEPL_LANG_MAP.get(target_language.lower(), target_language.upper()[:2])
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

    async def translate_batch(self, items: List[dict], target_language: str = "Swedish", context_lines: List[dict] = None, show_title: str = "") -> List[dict]:
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
        target_language: str = "Swedish",
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

        processed_count = 0
        context_window_size = 5
        previous_context = []

        for batch_idx, (start_idx, chunk, payload) in enumerate(batches):
            end_idx = min(start_idx + len(payload), total_lines)

            all_empty = all(p["text"].strip() == "<i></i>" for p in payload)
            if all_empty:
                processed_count += len(payload)
                if job_id:
                    update_job(job_id, processed_lines=processed_count, current_batch=f"Lines {start_idx + 1}-{end_idx} / {total_lines}")
                continue

            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    if job_id:
                        update_job(job_id, current_batch=f"Translating lines {start_idx + 1}-{end_idx} of {total_lines}")

                    results = await self.translate_batch(
                        payload,
                        target_language=target_language,
                        context_lines=previous_context if previous_context else None,
                        show_title=show_title or ""
                    )
                    res_dict = {r["id"]: r["text"] for r in results if "id" in r and "text" in r}
                    
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
                                if "id" in r and "text" in r:
                                    res_dict[r["id"]] = r["text"]
                            
                            still_missing = [p["id"] for p in payload if p["id"] not in res_dict]
                            if not still_missing and job_id:
                                append_job_log(job_id, f"QA Success: Smart Recovery successfully translated the missing lines!")
                        except Exception as e:
                            logger.error(f"Smart recovery failed: {e}")
                            if job_id:
                                append_job_log(job_id, f"QA Warning: Smart recovery failed ({e}). Some lines will remain original.")
                    # ----------------------------------------

                    for p in payload:
                        idx = p["id"]
                        if idx in res_dict:
                            translated_subs[idx].content = res_dict[idx]

                    processed_count += len(payload)
                    if job_id:
                        pct = int((processed_count / total_lines) * 100) if total_lines > 0 else 100
                        update_job(job_id, processed_lines=processed_count, current_batch=f"{pct}% ({processed_count}/{total_lines} lines)")
                        append_job_log(job_id, f"Progress {pct}%: Translated lines {start_idx + 1} to {end_idx} ({target_language})")

                    # Build context for next batch from last N translated lines
                    context_candidates = []
                    for p in payload[-context_window_size:]:
                        idx = p["id"]
                        if idx in res_dict:
                            context_candidates.append({"id": idx, "original": p["text"], "translated": res_dict[idx]})
                    previous_context = context_candidates

                    break
                except Exception as err:
                    retry_count += 1
                    err_str = str(err).lower()
                    if "429" in err_str or "503" in err_str or "504" in err_str or "quota" in err_str or "overloaded" in err_str:
                        backoff = [5, 15, 30][min(retry_count - 1, 2)]
                    else:
                        backoff = 2 * retry_count

                    logger.warning(f"Batch {start_idx + 1}-{end_idx} retry {retry_count}/{max_retries} (Waiting {backoff}s): {err}")
                    if job_id:
                        append_job_log(job_id, f"Notice: Retrying lines {start_idx + 1}-{end_idx} in {backoff}s (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(backoff)
            else:
                logger.error(f"Batch {start_idx} failed all retries.")
                if job_id:
                    append_job_log(job_id, f"Warning: Lines {start_idx + 1}-{end_idx} could not be translated. Keeping original text. (QA Recovery will attempt to fix this later)")

        return translated_subs

