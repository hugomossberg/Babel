import dataclasses
import re
from typing import Optional, Dict, List, Any
from app.core.db import get_setting

@dataclasses.dataclass(frozen=True)
class ModelCapabilities:
    structured_output: bool
    json_object: bool
    temperature: bool
    thinking_control: bool
    semantic_audit: bool
    entity_verification: bool
    max_output_tokens: Optional[int]
    native_output_config: bool = False  # Anthropic output_config.format (claude-sonnet-4.5+, claude-opus-4.1+)
    temperature_requires_no_reasoning: bool = False  # True for models where temperature is rejected unless reasoning_effort == "none"
    default_reasoning_effort: Optional[str] = None  # Documented API default reasoning effort (e.g. "medium", "none")

@dataclasses.dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    general_llm: bool
    default_model: str

DEFAULT_MODELS: Dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "openrouter": "google/gemini-3.7-flash",
    "deepseek": "deepseek-flash",
    "custom": "default",
    "ollama": "llama3",
    "deepl": "prefer_quality_optimized",
}

# Offline suggestions reviewed 2026-11-02. These never imply account availability.
# Official sources and lifecycle decisions: docs/model-discovery.md.
PROVIDER_FALLBACK_CATALOGS: Dict[str, list[Dict[str, str]]] = {
    "gemini": [
        {"id": "gemini-3.8-flash", "name": "gemini-3.8-flash"},
        {"id": "gemini-3.6-flash", "name": "gemini-3.6-flash"},
        {"id": "gemini-3.5-flash-lite", "name": "gemini-3.5-flash-lite"},
        {"id": "gemini-3.1-flash-lite", "name": "gemini-3.1-flash-lite"},
    ],
    "openai": [
        {"id": "gpt-6.1-sol", "name": "gpt-6.1-sol"},
        {"id": "gpt-6-luna", "name": "gpt-6-luna"},
        {"id": "gpt-5.6-sol", "name": "gpt-5.6-sol"},
        {"id": "gpt-5.6-terra", "name": "gpt-5.6-terra"},
        {"id": "gpt-5.6-luna", "name": "gpt-5.6-luna"},
        {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
        {"id": "gpt-4o", "name": "gpt-4o"},
        {"id": "gpt-4.1-mini", "name": "gpt-4.1-mini"},
        {"id": "gpt-4.1", "name": "gpt-4.1"},
        {"id": "o3-mini", "name": "o3-mini"},
    ],
    "anthropic": [
        {"id": "claude-sonnet-5-5", "name": "claude-sonnet-5-5"},
        {"id": "claude-haiku-5-5", "name": "claude-haiku-5-5"},
        {"id": "claude-sonnet-5", "name": "claude-sonnet-5"},
        {"id": "claude-opus-5", "name": "claude-opus-5"},
        {"id": "claude-haiku-4-5-20251001", "name": "claude-haiku-4-5-20251001"},
        {"id": "claude-fable-5", "name": "claude-fable-5"},
    ],
    "openrouter": [
        {"id": "google/gemini-3.7-flash", "name": "Google: Gemini 3.7 Flash"},
        {"id": "google/gemini-3.5-flash", "name": "Google: Gemini 3.5 Flash"},
        {"id": "anthropic/claude-sonnet-5", "name": "Anthropic: Claude Sonnet 5"},
        {"id": "openai/gpt-4o-mini", "name": "OpenAI: GPT-4o-mini"},
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
        {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek: DeepSeek V4 Flash"},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta: Llama 3.3 70B Instruct"},
        {"id": "qwen/qwen-2.5-72b-instruct", "name": "Qwen: Qwen 2.5 72B Instruct"},
    ],
    "deepseek": [
        {"id": "deepseek-flash", "name": "deepseek-flash"},
        {"id": "deepseek-v4-pro", "name": "deepseek-v4-pro"},
    ],
    "custom": [
        {"id": "default", "name": "default"},
    ],
    "ollama": [
        {"id": "llama3", "name": "llama3"},
        {"id": "llama3.1", "name": "llama3.1"},
        {"id": "mistral", "name": "mistral"},
        {"id": "qwen2.5", "name": "qwen2.5"},
    ],
    "deepl": [
        {"id": "prefer_quality_optimized", "name": "Prefer Quality Optimized (Recommended)"},
        {"id": "quality_optimized", "name": "Quality Optimized"},
        {"id": "latency_optimized", "name": "Latency Optimized"},
    ],
}

def get_default_model(provider: str) -> str:
    prov = normalize_provider(provider)
    return DEFAULT_MODELS.get(prov, "")

def get_provider_catalog(provider: str) -> list[Dict[str, str]]:
    prov = normalize_provider(provider)
    return [dict(m) for m in PROVIDER_FALLBACK_CATALOGS.get(prov, [])]


def _accepts_text(model: Dict[str, Any]) -> bool:
    for field in ("input_modalities", "output_modalities"):
        modalities = model.get(field)
        if isinstance(modalities, list) and "text" not in modalities:
            return False
    endpoints = model.get("supported_endpoints")
    if isinstance(endpoints, list) and not any(e in endpoints for e in ("chat/completions", "/v1/chat/completions", "chat_completions")):
        return False
    return model.get("lifecycle", model.get("status")) not in ("retired", "deprecated")


def filter_gemini_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter Gemini models from live API discovery:
    - Keep models supporting 'generateContent'.
    - Exclude non-text/specialized categories: image (imagen, veo), audio/music (lyria, tts, speech),
      embeddings, computer use, deep research, antigravity, robotics, etc.
    - Preserves exact model ID (without 'models/' prefix).
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    non_text_keywords = (
        "embedding", "embed", "imagen", "veo", "lyria", "music", "audio",
        "speech", "tts", "whisper", "transcri", "deep-research", "computer-use",
        "antigravity", "aqa", "robotics", "image", "nano-banana", "live"
    )
    for m in raw_models:
        name = m.get("name", "").removeprefix("models/").strip()
        if not name or name in seen:
            continue
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        lower_name = name.lower()
        lower_display = str(m.get("displayName", "")).lower()
        if any(kw in lower_name or kw in lower_display for kw in non_text_keywords):
            continue
        label = name + " — deprecated alias" if name in ("gemini-3.5-flash", "gemini-3.7-flash") else name
        discovered.append({"id": name, "name": label})
        seen.add(name)
    return discovered


def filter_openai_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter OpenAI models from live API discovery:
    - Keep text/chat/reasoning model families (gpt-, o1, o3, o4, chatgpt-).
    - Exclude non-text modalities & specialized endpoints: transcription/whisper, TTS, image (dall-e),
      embeddings, moderation, audio/realtime, computer-use, deep-research, instruct/davinci/babbage, etc.
    - Exclude fine-tuned models (ft:) and dated snapshot aliases when base exists.
    - Preserves exact model ID.
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    non_text_keywords = (
        "audio", "realtime", "tts", "transcription", "whisper", "moderation",
        "embedding", "dall-e", "instruct", "davinci", "babbage", "curie", "ada",
        "search", "computer-use", "deep-research", "image", "codex"
    )
    for m in raw_models:
        mid = m.get("id", "").strip()
        if not mid or mid in seen:
            continue
        lower_id = mid.lower()
        if not re.match(r"^(gpt-|o[134](?:-|$)|chatgpt-)", lower_id):
            continue
        if lower_id.startswith("ft:") or ":ft-" in lower_id:
            continue
        if any(kw in lower_id for kw in non_text_keywords):
            continue
        if re.search(r"(?:^o[134]|^gpt-).*?-pro(?:-|$)", lower_id):
            continue  # Pro/research/Codex endpoints require Responses, not Chat Completions.
        if lower_id == "o1-mini" or lower_id.startswith("o1-mini-") or lower_id.startswith("o1-preview"):
            continue  # Retired from the OpenAI API in 2025.
        # Prefer the alias when listed too; retain standalone snapshots and exact IDs.
        alias = re.sub(r"-\d{4}(?:-\d{2}-\d{2})?$", "", mid)
        if alias != mid and any(item.get("id") == alias for item in raw_models):
            continue
        discovered.append({"id": mid, "name": mid})
        seen.add(mid)
    return discovered


def filter_openrouter_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter OpenRouter models from live API discovery:
    - Filter out batch endpoints/models (:batch or (batch)).
    - Filter out non-text modalities using architecture metadata (output_modalities must include text).
    - Filter out non-text specialized keywords (image gen, whisper/audio, embedding, moderation, etc.).
    - Clean display name and deduplicate by model ID.
    - Preserves exact model ID (e.g. 'google/gemini-3.7-flash', 'anthropic/claude-3.5-sonnet').
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    non_text_keywords = (
        "embedding", "embed", "flux", "stable-diffusion", "midjourney", "dall-e",
        "imagen", "sdxl", "whisper", "transcri", "tts", "speech", "bark",
        "rerank", "moderation", "computer-use", "deep-research"
    )
    for m in raw_models:
        mid = m.get("id", "").strip()
        if not mid or mid in seen:
            continue
        if not _accepts_text(m):
            continue
        raw_name = m.get("name", "").strip() or mid
        lower_id = mid.lower()
        lower_name = raw_name.lower()
        if ":batch" in lower_id or "(batch)" in lower_name or "(batch)" in lower_id:
            continue
        arch = m.get("architecture")
        if isinstance(arch, dict):
            out_mods = arch.get("output_modalities")
            if isinstance(out_mods, list) and out_mods and "text" not in out_mods:
                continue
            mod = arch.get("modality", "")
            if isinstance(mod, str) and mod:
                if "->" in mod:
                    target = mod.split("->")[-1].strip().lower()
                    if "text" not in target:
                        continue
                elif mod.lower() in ["image", "embedding", "audio"]:
                    continue
            in_mods = arch.get("input_modalities")
            if isinstance(in_mods, list) and in_mods and "text" not in in_mods:
                continue
        if any(kw in lower_id or kw in lower_name for kw in non_text_keywords):
            continue
        clean_name = raw_name.replace("(batch)", "").replace("(Batch)", "").strip()
        discovered.append({"id": mid, "name": clean_name or mid})
        seen.add(mid)
    return discovered


def filter_anthropic_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter Anthropic models from live API discovery:
    - Filter non-text/specialized keywords if any.
    - Preserves exact model ID and uses clean display_name when available.
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    non_text_keywords = ("embedding", "embed", "image", "tts", "audio", "speech", "transcri", "moderation")
    for m in raw_models:
        mid = m.get("id", "").strip()
        if not mid or mid in seen:
            continue
        if m.get("lifecycle", m.get("status")) in ("retired", "deprecated"):
            continue
        # Old API versions omit lifecycle. These families are retired on Claude API.
        if mid.startswith(("claude-3-5-", "claude-3-7-", "claude-3-haiku-", "claude-opus-4-1-", "claude-opus-4-2025", "claude-sonnet-4-2025")):
            continue
        display_name = m.get("display_name", "").strip() or mid
        if any(kw in mid.lower() or kw in display_name.lower() for kw in non_text_keywords):
            continue
        discovered.append({"id": mid, "name": display_name})
        seen.add(mid)
    return discovered


def filter_ollama_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter Ollama models from tags endpoint:
    - Avoid aggressive filtering to preserve custom local models.
    - Filter obvious embedding-only models.
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    for m in raw_models:
        name = m.get("name", "").strip()
        if not name or name in seen:
            continue
        family = name.rsplit("/", 1)[-1].lower()
        if family.startswith(("nomic-embed", "mxbai-embed", "snowflake-arctic-embed", "bge-", "all-minilm", "gte-", "embeddinggemma", "qwen3-embedding", "granite-embedding")):
            continue
        discovered.append({"id": name, "name": name})
        seen.add(name)
    return discovered


def filter_custom_models(raw_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter Custom OpenAI-compatible models from endpoint:
    - Non-restrictive to allow user-defined endpoints and models.
    """
    discovered: List[Dict[str, str]] = []
    seen = set()
    for m in raw_models:
        mid = m.get("id", "").strip()
        if not mid or mid in seen:
            continue
        if not _accepts_text(m):
            continue
        discovered.append({"id": mid, "name": mid})
        seen.add(mid)
    return discovered

PROVIDERS: Dict[str, ProviderSpec] = {
    "gemini": ProviderSpec("gemini", "Gemini", True, DEFAULT_MODELS["gemini"]),
    "openai": ProviderSpec("openai", "OpenAI", True, DEFAULT_MODELS["openai"]),
    "anthropic": ProviderSpec("anthropic", "Anthropic", True, DEFAULT_MODELS["anthropic"]),
    "openrouter": ProviderSpec("openrouter", "OpenRouter", True, DEFAULT_MODELS["openrouter"]),
    "deepseek": ProviderSpec("deepseek", "DeepSeek", True, DEFAULT_MODELS["deepseek"]),
    "custom": ProviderSpec("custom", "Custom OpenAI", True, DEFAULT_MODELS["custom"]),
    "ollama": ProviderSpec("ollama", "Ollama", True, DEFAULT_MODELS["ollama"]),
    "deepl": ProviderSpec("deepl", "DeepL", False, DEFAULT_MODELS["deepl"]),
}

def normalize_provider(provider: str) -> str:
    provider = provider.lower().strip()
    if provider == "custom_openai":
        return "custom"
    if provider == "localai":
        return "ollama"
    return provider

def get_provider_spec(provider: str) -> ProviderSpec:
    provider = normalize_provider(provider)
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported AI provider: {provider}")
    return PROVIDERS[provider]

def provider_label(provider: str) -> str:
    return get_provider_spec(provider).label

def format_engine(provider: str, model: str) -> str:
    label = provider_label(provider)
    if model:
        return f"{label} ({model})"
    return label

@dataclasses.dataclass(frozen=True)
class ProviderContext:
    provider: str
    model: str
    
    @property
    def label(self) -> str:
        return provider_label(self.provider)
        
    @property
    def engine_label(self) -> str:
        return format_engine(self.provider, self.model)

def get_model_capabilities(provider: str, model: str) -> ModelCapabilities:
    raw_provider = (provider or "").lower().strip()
    if raw_provider == "localai":
        # LocalAI endpoints may reuse Ollama transport, but capabilities are unverified/conservative
        return ModelCapabilities(
            structured_output=False,
            json_object=False,
            temperature=False,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=None
        )
    provider = normalize_provider(provider)
    model = model.lower().strip() if model else ""
    
    if provider == "gemini":
        # Leave Gemini 3.x sampling at the provider-recommended default.
        # Legacy Gemini 1.5 / 2.0 / 2.5 series models accept temperature.
        # Unknown/future Gemini models use conservative default (temperature=False).
        temp = False
        if any(prefix in model for prefix in ["gemini-1.5", "gemini-2.0", "gemini-2.5"]):
            temp = True
        return ModelCapabilities(
            structured_output=True,
            json_object=True,
            temperature=temp,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=8192
        )
    elif provider == "anthropic":
        # Use provider-default sampling for modern Claude models; native JSON on known families.
        # Claude 3.5 and earlier models support temperature but not native output_config.format
        # Unknown Anthropic models use conservative default (temperature=False, native_output_config=False)
        _native_oc = (
            "claude-sonnet-5" in model
            or "claude-opus-5" in model
            or "claude-fable-5" in model
            or "claude-haiku-5" in model
            or "claude-sonnet-4-6" in model
            or any(prefix in model for prefix in ("claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"))
            or "claude-haiku-4-5" in model
            or "claude-sonnet-4-5" in model
            or "claude-sonnet-4.5" in model
            or "claude-opus-4-1" in model
            or "claude-opus-4.1" in model
        )
        temp = False
        if any(prefix in model for prefix in ["claude-3-5", "claude-3-opus", "claude-3-haiku", "claude-3-sonnet", "claude-2"]):
            temp = True
        return ModelCapabilities(
            structured_output=True,
            json_object=False,
            temperature=temp,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=8192,
            native_output_config=_native_oc,
        )
    elif provider == "deepseek":
        # Known DeepSeek chat models support temperature and thinking control
        # DeepSeek reasoner models (r1) do not support temperature
        # Unknown DeepSeek models use conservative default
        if model in ("deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat"):
            return ModelCapabilities(
                structured_output=False,
                json_object=True,
                temperature=True,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=8192
            )
        elif "reasoner" in model or "r1" in model:
            return ModelCapabilities(
                structured_output=False,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=8192
            )
        # Conservative fallback for unknown deepseek model
        return ModelCapabilities(
            structured_output=False,
            json_object=True,
            temperature=False,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=8192
        )
    elif provider == "openai":
        if re.fullmatch(r"gpt-(6(?:-astra|-sol|-luna)|6\.1(?:-sol|-terra))(?:-\d{4}-\d{2}-\d{2})?", model):
            supports_none = model.startswith(("gpt-6.1-terra", "gpt-6-sol", "gpt-6-luna"))
            return ModelCapabilities(
                structured_output=True, json_object=True, temperature=False,
                thinking_control=True, semantic_audit=True, entity_verification=True,
                max_output_tokens=8192,
                temperature_requires_no_reasoning=supports_none,
            )
        # 1. Reasoning o-series models (o1, o1-mini, o1-preview, o3, o3-mini, o4, o4-mini): reject custom temperature
        is_o_series = bool(re.match(r"^o(1|3|4)(-mini|-preview)?(-\d{4}-\d{2}-\d{2})?$", model) or model in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4", "o4-mini"])

        # 2. GPT-5.6 flagship family: default reasoning = "medium", supports reasoning "none"
        is_gpt_5_6 = bool(model in ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] or re.match(r"^gpt-5\.6(-(sol|terra|luna))?(-\d{4}-\d{2}-\d{2})?$", model))

        # 3. GPT-5.5: default reasoning = "medium", supports reasoning "none"
        is_gpt_5_5 = bool(model == "gpt-5.5" or re.match(r"^gpt-5\.5(-preview)?(-\d{4}-\d{2}-\d{2})?$", model))

        # 4. GPT-5.1, GPT-5.2, GPT-5.4: default reasoning = "none", supports reasoning "none" (temperature allowed by default)
        is_gpt_5_default_none = bool(
            model in ["gpt-5.1", "gpt-5.2", "gpt-5.4"]
            or re.match(r"^gpt-5\.(1|2|4)(-preview)?(-\d{4}-\d{2}-\d{2})?$", model)
        )

        # 5. GPT-5.3-codex: supports reasoning (low, medium, high, xhigh) but does NOT support reasoning "none"
        is_gpt_5_3_codex = bool(model == "gpt-5.3-codex" or model.startswith("gpt-5.3-codex"))

        # 6. GPT-5.3-chat-latest: deprecated Instant/Chat model, non-reasoning, supports temperature and structured output
        is_gpt_5_3_chat = bool(model == "gpt-5.3-chat-latest" or model.startswith("gpt-5.3-chat-latest"))

        # 7. Early/base GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano): default reasoning = "medium", does NOT support temperature
        is_early_gpt_5 = bool(model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"] or re.match(r"^gpt-5(-mini|-nano|-preview)?(-\d{4}-\d{2}-\d{2})?$", model))

        # 8. Classical non-reasoning GPT models (gpt-4o, gpt-4o-mini, gpt-4.1 series, gpt-4-turbo, gpt-3.5-turbo, chatgpt-)
        is_classical_gpt = bool(
            re.match(r"^gpt-4o(-mini)?(-\d{4}-\d{2}-\d{2})?$", model)
            or re.match(r"^gpt-4\.1(-mini|-nano)?(-\d{4}-\d{2}-\d{2})?$", model)
            or re.match(r"^gpt-4(-turbo)?(-preview)?(-\d{4}-\d{2}-\d{2})?$", model)
            or re.match(r"^gpt-3\.5(-turbo)?(-\d{4}-\d{2}-\d{2})?$", model)
            or model.startswith("chatgpt-")
        )

        if is_o_series:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort="medium",
            )
        elif is_gpt_5_6:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=True,
                default_reasoning_effort="medium",
            )
        elif is_gpt_5_5:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=True,
                default_reasoning_effort="medium",
            )
        elif is_gpt_5_default_none:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=True,
                default_reasoning_effort="none",
            )
        elif is_gpt_5_3_codex:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
        elif is_gpt_5_3_chat:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=True,
                thinking_control=False,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
        elif is_early_gpt_5:
            return ModelCapabilities(
                structured_output=True,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort="medium",
            )
        elif is_classical_gpt:
            # JSON mode predates strict schemas. Do not send json_schema to
            # GPT-3.5, GPT-4 Turbo or the first GPT-4o snapshot (official API).
            schema_supported = (
                model.startswith(("gpt-4o-mini", "gpt-4.1"))
                or model in ("gpt-4o", "chatgpt-4o-latest")
                or bool(re.fullmatch(r"gpt-4o-\d{4}-\d{2}-\d{2}", model)
                        and model >= "gpt-4o-2024-08-06")
            )
            json_supported = schema_supported or model.startswith(("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-"))
            return ModelCapabilities(
                structured_output=schema_supported,
                json_object=json_supported,
                temperature=True,
                thinking_control=False,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
        else:
            # Conservative fallback for unknown/future OpenAI models
            return ModelCapabilities(
                structured_output=False,
                json_object=False,
                temperature=False,
                thinking_control=False,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=4096,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
    elif provider == "openrouter":
        if model.startswith("openai/gpt-6"):
            return get_model_capabilities("openai", model.removeprefix("openai/"))
        # Specific known OpenAI models via OpenRouter
        if model in [
            "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.6-luna",
            "openai/gpt-5.6", "openai/gpt-5.5"
        ]:
            return ModelCapabilities(
                structured_output=False,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=None,
                temperature_requires_no_reasoning=True,
                default_reasoning_effort="medium",
            )
        elif model in ["openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-5.4"]:
            return ModelCapabilities(
                structured_output=False,
                json_object=True,
                temperature=False,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=None,
                temperature_requires_no_reasoning=True,
                default_reasoning_effort="none",
            )
        elif any(m in model for m in ["openai/gpt-4o", "meta-llama/", "mistralai/", "qwen/", "anthropic/claude-3.5-sonnet"]):
            temp = True
            json_obj = any(m in model for m in ["openai/gpt-4o"])
            return ModelCapabilities(
                structured_output=False,
                json_object=json_obj,
                temperature=temp,
                thinking_control=False,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=None,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
        elif any(m in model for m in ["deepseek/deepseek-v4-flash"]):
            return ModelCapabilities(
                structured_output=False,
                json_object=True,
                temperature=True,
                thinking_control=True,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=None,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
        else:
            # Conservative fallback for unknown/future OpenRouter models
            return ModelCapabilities(
                structured_output=False,
                json_object=False,
                temperature=False,
                thinking_control=False,
                semantic_audit=True,
                entity_verification=True,
                max_output_tokens=None,
                temperature_requires_no_reasoning=False,
                default_reasoning_effort=None,
            )
    elif provider == "custom":
        # Conservative defaults for custom OpenAI endpoints: no optional params
        return ModelCapabilities(
            structured_output=False,
            json_object=False,
            temperature=False,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=None
        )
    elif provider == "ollama":
        is_known_ollama = any(pref in model for pref in ["llama3", "mistral", "qwen", "phi", "gemma"])
        return ModelCapabilities(
            structured_output=False,
            json_object=True,
            temperature=True if is_known_ollama else False,
            thinking_control=False,
            semantic_audit=True,
            entity_verification=True,
            max_output_tokens=None
        )
    elif provider == "deepl":
        return ModelCapabilities(
            structured_output=False,
            json_object=False,
            temperature=False,
            thinking_control=False,
            semantic_audit=False,
            entity_verification=False,
            max_output_tokens=None
        )
        
    # Conservative fallback for unknown providers
    return ModelCapabilities(
        structured_output=False,
        json_object=False,
        temperature=False,
        thinking_control=False,
        semantic_audit=False,
        entity_verification=False,
        max_output_tokens=None
    )

def context_from_settings(provider: Optional[str] = None, *, escalation: bool = False) -> ProviderContext:
    if escalation:
        esc_enabled = get_setting("escalate_to_pro", "false").lower() == "true"
        esc_provider = get_setting("escalation_provider", "none").lower().strip()
        if not esc_enabled or not esc_provider or esc_provider == "none":
            return context_from_settings()

        provider = normalize_provider(esc_provider)
        spec = get_provider_spec(provider)
        esc_model = get_setting("escalation_model", "").strip()
        if esc_model:
            model = esc_model
        else:
            model = get_setting(f"{provider}_model", spec.default_model)
        return ProviderContext(provider=provider, model=model)

    if not provider:
        provider = get_setting("ai_provider", "gemini")
    
    provider = normalize_provider(provider)
    spec = get_provider_spec(provider)
    
    if provider == "gemini":
        model = get_setting("gemini_model", spec.default_model)
    elif provider == "openai":
        model = get_setting("openai_model", spec.default_model)
    elif provider == "anthropic":
        model = get_setting("anthropic_model", spec.default_model)
    elif provider == "openrouter":
        model = get_setting("openrouter_model", spec.default_model)
    elif provider == "deepseek":
        model = get_setting("deepseek_model", spec.default_model)
    elif provider == "custom":
        model = get_setting("custom_openai_model", spec.default_model)
    elif provider == "ollama":
        model = get_setting("ollama_model", spec.default_model)
    elif provider == "deepl":
        model = get_setting("deepl_model_type", spec.default_model)
    else:
        model = spec.default_model
        
    return ProviderContext(provider=provider, model=model)

def resolve_job_provider_context(job_id: int, *, escalation: bool = False) -> ProviderContext:
    from app.core.db import DB_PATH
    import sqlite3
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT primary_provider, primary_model, escalation_provider, escalation_model 
            FROM jobs WHERE id = ?
        ''', (job_id,))
        row = cursor.fetchone()
        
    if row:
        if escalation:
            # If escalation is requested but not configured on job, fallback to primary or settings
            prov = row["escalation_provider"] or row["primary_provider"]
            if prov:
                mod = row["escalation_model"] if row["escalation_provider"] else row["primary_model"]
                try:
                    return ProviderContext(provider=normalize_provider(prov), model=mod or "")
                except ValueError:
                    pass
        else:
            prov = row["primary_provider"]
            if prov:
                mod = row["primary_model"]
                try:
                    return ProviderContext(provider=normalize_provider(prov), model=mod or "")
                except ValueError:
                    pass
                    
    # Fallback to settings if job not pinned or provider unknown
    return context_from_settings(escalation=escalation)
