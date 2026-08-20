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
            item_text = m.group(2).encode().decode('unicode_escape')
            items.append({"id": item_id, "text": item_text})
        except Exception:
            pass

    if items:
        return items

    raise ValueError(f"Could not extract JSON translation array from response: {raw_text[:100]}...")


class SubtitleTranslator:
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

    async def translate_batch_gemini(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None) -> List[dict]:
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
        show_title = getattr(self, '_current_show_title', '') or ''
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

    async def translate_batch_openai(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None) -> List[dict]:
        client = self.get_openai_client()
        
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'
                
        prompt = f"Translate the following {len(items)} subtitle lines into {target_language}:{context_section}\n\n" + json.dumps(items, ensure_ascii=False)

        glossary = get_setting("glossary", "")
        show_title = getattr(self, '_current_show_title', '') or ''
        
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
        
        target_lang_code = "SV" if target_language.lower() in ["swedish", "sv"] else target_language.upper()[:2]
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
            return [{"id": items[i]["id"], "text": translations[i]["text"]} for i in range(len(items))]

    async def translate_batch_ollama(self, items: List[dict], target_language: str, model_name: str, context_lines: List[dict] = None) -> List[dict]:
        ollama_url = get_setting("ollama_url", "http://localhost:11434").rstrip("/")
        
        context_section = ""
        if context_lines:
            context_section = "\n\nCONTEXT from previous batch (DO NOT translate these, use them only for tone and consistency):\n"
            for ctx in context_lines:
                context_section += f'  Original: "{ctx["original"]}" → Translated: "{ctx["translated"]}"\n'

        glossary = get_setting("glossary", "")
        show_title = getattr(self, '_current_show_title', '') or ''
        
        prompt = f"{get_system_instruction(target_language, glossary=glossary, show_title=show_title)}\n\nTranslate the following JSON list into {target_language}:{context_section}\n{json.dumps(items, ensure_ascii=False)}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name or "llama3", "prompt": prompt, "format": "json", "stream": False}
            )
            resp.raise_for_status()
            data = resp.json()
            return extract_json_safely(data.get("response", "{}"))

    async def translate_batch(self, items: List[dict], target_language: str = "Swedish", context_lines: List[dict] = None) -> List[dict]:
        provider = get_setting("ai_provider", "gemini").lower()
        if provider == "openai":
            model = get_setting("openai_model", "gpt-4o-mini")
            return await self.translate_batch_openai(items, target_language, model, context_lines=context_lines)
        elif provider == "deepl":
            return await self.translate_batch_deepl(items, target_language, context_lines=context_lines)
        elif provider in ["ollama", "localai"]:
            model = get_setting("ollama_model", "llama3")
            return await self.translate_batch_ollama(items, target_language, model, context_lines=context_lines)
        else:
            model = get_setting("gemini_model", "gemini-3.5-flash-lite")
            return await self.translate_batch_gemini(items, target_language, model, context_lines=context_lines)

    async def translate_srt_content(
        self,
        subs: List[srt.Subtitle],
        target_language: str = "Swedish",
        batch_size: int = 50,
        job_id: Optional[int] = None,
        show_title: Optional[str] = None
    ) -> List[srt.Subtitle]:
        self._current_show_title = show_title
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
                        context_lines=previous_context if previous_context else None
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
                                context_lines=previous_context if previous_context else None
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
                    logger.warning(f"Batch {start_idx + 1}-{end_idx} retry {retry_count}/{max_retries}: {err}")
                    if job_id:
                        append_job_log(job_id, f"Notice: Retrying lines {start_idx + 1}-{end_idx} (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(1.5 * retry_count)
            else:
                logger.error(f"Batch {start_idx} failed all retries.")
                if job_id:
                    append_job_log(job_id, f"Warning: Lines {start_idx + 1}-{end_idx} could not be translated. Keeping original text.")

        return translated_subs

