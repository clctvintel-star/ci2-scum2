def call_gemini(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None

    for _ in range(retry_limit):
        try:
            r = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "max_output_tokens": 800,
                    "response_mime_type": "application/json",
                },
            )

            # 1) clean text shortcut
            if hasattr(r, "text") and r.text:
                return r.text

            # 2) walk candidates/parts safely
            if hasattr(r, "candidates") and r.candidates:
                texts = []
                for cand in r.candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts:
                        for part in parts:
                            txt = getattr(part, "text", None)
                            if txt:
                                texts.append(txt)
                if texts:
                    return "\n".join(texts).strip()

            # 3) last resort: stringify whole response for debug
            raise ValueError(f"No valid text in Gemini response: {r}")

        except Exception as e:
            last_err = e
            time.sleep(1)

    print(f"⚠️ Gemini empty response (likely MAX_TOKENS): {last_err}")
    return f"__GEMINI_EMPTY__::{last_err}"

def call_model(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    if model_name.startswith("claude"):
        return call_anthropic(
            anthropic_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gpt"):
        return call_openai(
            openai_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gemini"):
        return call_gemini(
            gemini_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    raise ValueError(f"Unsupported model family: {model_name}")


def parse_sentiment_response(text: Any) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    if isinstance(text, dict):
        payload = {str(k).lower(): v for k, v in text.items()}
    else:
        payload = extract_json_payload(text) or {}

    if not payload:
        return None, None, None, "parse_failure"

    sentiment = payload.get("sentiment")
    confidence = payload.get("confidence")
    explanation = payload.get("explanation") or payload.get("reason") or payload.get("rationale")

    sentiment = clamp_float(sentiment, -1.0, 1.0)
    confidence = clamp_float(confidence, 0.0, 1.0)
    explanation = safe_text(explanation) or None

    if sentiment is None and confidence == 0.0:
        sentiment = 0.0
        confidence = 0.10
        return sentiment, confidence, explanation, "neutralized_abstention"

    if sentiment is None or confidence is None:
        return None, None, explanation, "parse_failure"

    return sentiment, confidence, explanation, "valid"


def call_and_parse_sentiment(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
    semantic_retry_limit: int = 3,
) -> Tuple[Optional[float], Optional[float], Optional[str], str, str]:
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        base_prompt = prompt if len(prompt) < 4000 else prompt[:4000]

        effective_prompt = ""
        if attempt == 0:
            effective_prompt = (
                "Return ONLY valid JSON. No prose. No explanation outside JSON.\n"
                + '{"sentiment": number, "confidence": number, "explanation": "..."}\n\n'
                + base_prompt
            )
        elif attempt == 1:
            effective_prompt = (
                "STRICT: Output MUST be valid JSON. No text before or after.\n"
                + '{"sentiment": number, "confidence": number, "explanation": "..."}\n\n'
                + base_prompt
            )
        else:
            effective_prompt = (
                "FINAL ATTEMPT. If you do not return valid JSON, this response will be discarded.\n"
                + '{"sentiment": number, "confidence": number, "explanation": "..."}\n\n'
                + prompt[:2500]  # Trim more aggressively for the final attempt
            )

        raw = call_model(
            model_name=model_name,
            prompt=effective_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

        last_raw = raw if isinstance(raw, str) else ""

        cleaned = last_raw.strip()

        # Handle the Gemini empty sentinel failure
        if cleaned.startswith("__GEMINI_EMPTY__"):
            return None, None, None, last_raw, "model_failure"

        cleaned = re.sub(
            r"^.*?json.*?:",  # Hardened regex for preamble
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            continue

        payload = extract_json_payload(cleaned)
        if not payload:
            continue

        # Pass the extracted payload directly to parse_sentiment_response
        sentiment, confidence, reason, status = parse_sentiment_response(payload)

        if status in {"valid", "neutralized_abstention"}:
            return sentiment, confidence, reason, last_raw, status

    return None, None, None, last_raw, "parse_failure"
