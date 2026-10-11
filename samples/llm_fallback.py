"""Excerpt from Clipper's LLM client (backend/app/pipeline/llm.py): the entry point that
retries across Gemini models and API keys, cools down overloaded models, and falls back to
free OpenRouter models. Helpers and settings it references live in the private repo."""

def complete_json(
    system: str,
    prompt: str,
    *,
    max_tokens: int = 8192,
    temperature: float = 0.4,
    validate: Callable[[dict], T] | None = None,
    model_fallback: bool = True,
    full_size_fallback: bool = False,
    task: str | None = None,
    images: list[bytes] | None = None,
) -> Any:
    """`_complete_json`, then the free OpenRouter models when Gemini is unavailable
    (text-only calls, and only when OPENROUTER_API_KEY is set)."""
    try:
        return _complete_json(
            system, prompt, max_tokens=max_tokens, temperature=temperature,
            validate=validate, model_fallback=model_fallback,
            full_size_fallback=full_size_fallback, task=task, images=images,
        )
    except LLMUnavailable as exc:
        s = get_settings()
        if images or s.analysis_provider != "gemini" or not s.openrouter_api_key:
            raise
        return _complete_via_openrouter(
            system, prompt, max_tokens=max_tokens, temperature=temperature,
            validate=validate, gemini_failure=exc,
        )


def _complete_json(
    system: str,
    prompt: str,
    *,
    max_tokens: int = 8192,
    temperature: float = 0.4,
    validate: Callable[[dict], T] | None = None,
    model_fallback: bool = True,
    full_size_fallback: bool = False,
    task: str | None = None,
    images: list[bytes] | None = None,
) -> Any:
    """Run one LLM completion expecting a strict-JSON object back.

    Uses the configured provider. For Gemini, `model_fallback` tries the primary
    then the fallback model when a call raises OR (if `validate` is given) when
    the validator rejects the response - set it False when the caller runs its
    own retry loop, so retries don't multiply. Supports multi-key pools via
    `gemini_api_keys` and task-specific keys. Returns the raw dict, or
    `validate(dict)` when a validator is given.
    """
    s = get_settings()

    if s.analysis_provider == "claude":
        if not s.anthropic_api_key:
            raise LLMConfigError("ANALYSIS_PROVIDER=claude but ANTHROPIC_API_KEY is unset")
        models: list[str | None] = [None]
        keys: list[str] = []
    elif s.analysis_provider == "gemini":
        keys = s.gemini_keys(task=task)
        if not keys:
            raise LLMConfigError("GEMINI_API_KEY (or GEMINI_API_KEYS) is unset")
        if _all_keys_dead(keys):
            raise LLMConfigError(
                "the provider rejected the API key (every key in the pool); fix "
                "GEMINI_API_KEY / GEMINI_API_KEYS and restart the backend (a key "
                "re-enabled with the provider is retried within 30 minutes)"
            )
        # a task key (e.g. GEMINI_KEY_WORDTAGS) stays first so its quota is used
        # before the shared pool's
        pinned = keys[0] if task and keys != s.gemini_keys() else None
        dead_at_start = _dead_keys(keys)
        deadline_s = float(s.gemini_deadline_s)
        if s.openrouter_api_key and not images:
            deadline_s = min(deadline_s, float(s.openrouter_switch_after_s))
        deadline = time.monotonic() + deadline_s
        models = [s.gemini_model]
        # Quota is per model. Clip finding and ending picks lose real quality on a
        # Lite model, so those callers ask to try the other full-size models first.
        if model_fallback and full_size_fallback:
            models += [m for m in _ANALYSIS_FULL_MODELS if m not in models]
        if model_fallback and s.gemini_fallback_model and s.gemini_fallback_model != s.gemini_model:
            models.append(s.gemini_fallback_model)
        for extra in ("gemini-flash-latest", "gemini-3.1-flash-lite"):
            if model_fallback and extra not in models:
                models.append(extra)
        with _STATE_LOCK:
            now_ts = time.monotonic()
            active = [m for m in models if _MODEL_COOLDOWN.get(m, 0.0) <= now_ts]
        # a cooling model was just overloaded or throttled; only fall back to it
        # when every model is cooling, rather than paying its retry cost each call.
        models = active or models
    else:
        raise LLMConfigError(
            f"unsupported ANALYSIS_PROVIDER: {s.analysis_provider!r} (use 'gemini' or 'claude')"
        )

    errors: list[str] = []
    kinds: set[str] = set()
    auth_codes: list[int | None] = []

    if s.analysis_provider == "claude":
        for model in models:
            label = "claude-sonnet-4-5"
            try:
                payload = _call_once(
                    system, prompt, model, max_tokens=max_tokens, temperature=temperature,
                    images=images,
                )
                return validate(payload) if validate else payload
            except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                if isinstance(exc, AnswerRejected):
                    kinds.add("rejected")
                elif _is_transient_error(exc):
                    kinds.add("transient")
                elif _is_auth_error(exc):
                    kinds.add("auth")
                    auth_codes.append(_status_code(exc))
                else:
                    kinds.add("other")
    else:
        timed_out = False
        for model in models:
            ordered_keys = _pick_ordered_keys(keys, model, pinned)
            if not ordered_keys:
                # every key is dead or has used up its daily quota for this model
                errors.append(f"{model}: no usable key (rejected or daily quota used up)")
                kinds.add("transient")
                continue
            # why each key failed on this model: overload | quota | dead | transient | rejected | other
            model_fails: list[str] = []
            for key_idx, key in enumerate(ordered_keys):
                if _time_left(deadline) <= 0:
                    timed_out = True
                    break
                st = _get_key_state(key)
                has_spare = key_idx < len(ordered_keys) - 1
                label = f"{model}[{st.key_hint()}]"
                try:
                    payload = _call_once(
                        system,
                        prompt,
                        model,
                        api_key=key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        has_spare_keys=has_spare,
                        deadline=deadline,
                        images=images,
                    )
                    return validate(payload) if validate else payload
                except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                    errors.append(f"{label}: {type(exc).__name__}: {exc}")
                    if isinstance(exc, AnswerRejected):
                        kinds.add("rejected")
                        model_fails.append("rejected")
                        break
                    elif _is_transient_error(exc):
                        kinds.add("transient")
                        if _is_daily_quota_exhausted(exc):
                            model_fails.append("quota")
                        elif _is_overloaded_or_throttled(exc):
                            model_fails.append("overload")
                            if _status_code(exc) in (503, 529) and model_fails.count("overload") >= 2 and len(models) > 1:
                                break
                        else:
                            model_fails.append("transient")
                    elif _is_auth_error(exc):
                        _mark_key_dead(key)
                        kinds.add("dead_key")
                        auth_codes.append(_status_code(exc))
                        model_fails.append("dead")
                    else:
                        kinds.add("other")
                        model_fails.append("other")
            # only a 429/503/529 cools the model for other callers; a rejected
            # answer, a parse error or a bad key says nothing about its capacity
            if "overload" in model_fails and set(model_fails) <= {"overload", "quota", "dead"}:
                _record_cooldown(model, 180.0)
                logger.warning("LLM %s overloaded on every key; placing it on a 3m cooldown", model)
            if timed_out:
                break

        if timed_out:
            raise LLMUnavailable(
                f"Gemini didn't answer within {deadline_s / 60:.0f} minutes, try again later.\n"
                + "\n".join(errors)
            )
        if _all_keys_dead(keys):
            logger.warning("LLM auth failure on every key: %s", "\n".join(errors))
            codes = f", HTTP {auth_codes[0]}" if auth_codes else ""
            raise LLMConfigError(f"the provider rejected the API key{codes}")
        # some key is still live: a rejected key has left the pool and doesn't
        # decide the outcome
        if kinds == {"dead_key"} and dead_at_start - _dead_keys(keys):
            # every key this call tried was rejected, and a key it skipped as dead
            # came back (its sit-out ended mid-call). A retry tries it.
            raise LLMUnavailable("LLM completion failed.\n" + "\n".join(errors))
        kinds.discard("dead_key")

    msg = "LLM completion failed.\n" + "\n".join(errors)
    if kinds == {"auth"}:
        logger.warning("LLM auth failure across all models/keys: %s", msg)
        raise LLMConfigError(f"the provider rejected the API key, HTTP {auth_codes[0]}")
    if kinds == {"rejected"}:
        raise AnswerRejected(msg)
    # an outage decides the outcome even when another model's answer was rejected:
    # retrying later can still succeed
    if "transient" in kinds and kinds <= {"transient", "rejected"}:
        raise LLMUnavailable(msg)
    raise RuntimeError(msg)
