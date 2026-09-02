"""Cross-protocol orchestration: resolve upstream, wire, execute, recover."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import AppConfig, UpstreamConfig
from ..fc.obfuscation import deobfuscate_calls, obfuscate_tools
from ..fc.parse import parse_text_to_calls
from ..fc.policy import apply_policy
from ..fc.profiles import build_few_shot_block, detect_tool_profile, profile_instruction_block
from ..fc.recovery import build_retry_user_message, parse_with_recovery_hint
from ..models.canonical import CanonicalRequest, CanonicalResponse, Message, ToolCall
from ..stream.openai_sse import stream_native_passthrough, stream_prompt_fc
from ..stream.responses_sse import stream_responses_from_openai_chat
from ..upstream.router import UpstreamRouter
from ..util.ids import random_id
from ..util.metrics import GLOBAL_METRICS
from ..util.sse import done_frame, format_sse, iter_sse_events
from .wire import (
    anthropic_result_to_canonical,
    build_anthropic_wire,
    build_gemini_wire,
    build_openai_wire,
    encode_client_response,
    gemini_result_to_canonical,
    openai_result_to_canonical,
)


def _extra_instructions(req: CanonicalRequest, config: AppConfig) -> str:
    chunks: List[str] = []
    if config.features.enable_cli_profiles and req.tools:
        profile = detect_tool_profile(req.tools)
        chunks.append(profile_instruction_block(profile))
        few = build_few_shot_block(req.tools)
        if few:
            chunks.append(few)
    return "\n\n".join(chunks)


def _prepare_tools_for_prompt(req: CanonicalRequest, config: AppConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if config.features.obfuscate_tool_names and req.tools:
        tools, mapping = obfuscate_tools(req.tools)
        req.tools = tools
    return mapping


def _stream_headers(fc_mode: str) -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-ToolForge-FC-Mode": fc_mode,
    }


async def _stream_anthropic_native(line_iter: AsyncIterator[str], *, model: str) -> AsyncIterator[str]:
    """Pass-through Anthropic SSE with proper event framing and error safety."""
    try:
        async for event in iter_sse_events(line_iter):
            yield format_sse(event.data, event=event.event)
    except Exception as exc:  # noqa: BLE001
        yield format_sse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            },
            event="error",
        )
        yield format_sse({"type": "message_stop"}, event="message_stop")


async def _stream_anthropic_prompt(
    line_iter: AsyncIterator[str],
    *,
    req: CanonicalRequest,
    config: AppConfig,
) -> AsyncIterator[str]:
    """Parse prompt-FC from Anthropic text stream into tool_use SSE events."""
    text_parts: List[str] = []
    msg_id = f"msg_{random_id(16)}"
    yield format_sse(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        event="message_start",
    )
    try:
        async for event in iter_sse_events(line_iter):
            payload = event.json()
            if not payload:
                continue
            ptype = payload.get("type")
            if ptype == "content_block_delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text_parts.append(str(delta.get("text") or ""))
            elif ptype == "content_block_start":
                block = payload.get("content_block") or {}
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(str(block.get("text") or ""))
            # Also accept openai-compat style if upstream was remapped
            choices = payload.get("choices") or []
            if choices:
                delta = (choices[0] or {}).get("delta") or {}
                if delta.get("content"):
                    text_parts.append(str(delta.get("content")))
    except Exception as exc:  # noqa: BLE001
        yield format_sse(
            {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            event="error",
        )
        yield format_sse({"type": "message_stop"}, event="message_stop")
        return

    full = "".join(text_parts)
    calls = parse_text_to_calls(
        full,
        req.tools,
        protocol=config.features.inject_protocol,
        strip_think=config.features.strip_think_tags,
    )
    index = 0
    if calls:
        for call in calls:
            yield format_sse(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.id or f"toolu_{random_id(12)}",
                        "name": call.name,
                        "input": {},
                    },
                },
                event="content_block_start",
            )
            args = json.dumps(call.arguments or {}, ensure_ascii=False)
            yield format_sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                },
                event="content_block_delta",
            )
            yield format_sse({"type": "content_block_stop", "index": index}, event="content_block_stop")
            index += 1
        stop_reason = "tool_use"
    else:
        if full:
            yield format_sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            )
            yield format_sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": full},
                },
                event="content_block_delta",
            )
            yield format_sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
        stop_reason = "end_turn"
    yield format_sse(
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}},
        event="message_delta",
    )
    yield format_sse({"type": "message_stop"}, event="message_stop")


async def _stream_gemini(
    line_iter: AsyncIterator[str],
    *,
    req: CanonicalRequest,
    config: AppConfig,
) -> AsyncIterator[str]:
    """Incremental Gemini SSE: forward partial candidates when possible; final reassembled if prompt FC."""
    text_parts: List[str] = []
    native_calls: List[ToolCall] = []
    try:
        if req.fc_mode != "prompt":
            # Native: reframe each upstream event; patch model if needed
            async for event in iter_sse_events(line_iter):
                if event.is_done:
                    yield done_frame()
                    return
                payload = event.json()
                if payload is None:
                    if event.data:
                        yield format_sse(event.data)
                    continue
                yield format_sse(payload)
            yield done_frame()
            return

        async for event in iter_sse_events(line_iter):
            if event.is_done:
                break
            payload = event.json()
            if not payload:
                continue
            # OpenAI-compat upstream text
            choices = payload.get("choices") or []
            if choices:
                delta = (choices[0] or {}).get("delta") or {}
                if delta.get("content"):
                    text_parts.append(str(delta.get("content")))
                continue
            for cand in payload.get("candidates") or []:
                content = (cand or {}).get("content") or {}
                for part in content.get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        text_parts.append(str(part.get("text") or ""))
                    fc = part.get("functionCall") or part.get("function_call")
                    if isinstance(fc, dict):
                        args = fc.get("args") or {}
                        if not isinstance(args, dict):
                            args = {"value": args}
                        native_calls.append(
                            ToolCall(id="", name=str(fc.get("name") or ""), arguments=args)
                        )
    except Exception as exc:  # noqa: BLE001
        yield format_sse({"error": {"message": str(exc)}})
        yield done_frame()
        return

    full = "".join(text_parts)
    calls = list(native_calls)
    if req.fc_mode == "prompt":
        parsed = parse_text_to_calls(
            full,
            req.tools,
            protocol=config.features.inject_protocol,
            strip_think=config.features.strip_think_tags,
        )
        if parsed:
            calls = parsed
            full = ""
        else:
            import re as _re
            full = _re.sub(r"<[|\|] ?[|\|] ?(XYML|DSML)[|\|] ?[|\|] ?[^>]*>", "", full, flags=_re.IGNORECASE)
            full = _re.sub(r"<[|\|] ?[|\|] ?(XYML|DSML)[|\|] ?[|\|] ?[^>]*>", "", full, flags=_re.IGNORECASE)
            full = _re.sub(r"</[|\|] ?[|\|] ?(XYML|DSML)[|\|] ?[|\|] ?[^>]*>", "", full, flags=_re.IGNORECASE)
            full = _re.sub(r"</[|\|] ?[|\|] ?(XYML|DSML)[|\|] ?[|\|] ?[^>]*>", "", full, flags=_re.IGNORECASE)
    from ..convert import gemini_response

    yield format_sse(gemini_response(model=req.model, content=full, tool_calls=calls or None))
    yield done_frame()


async def _execute_once(
    req: CanonicalRequest,
    *,
    config: AppConfig,
    router: UpstreamRouter,
    upstream_cfg: UpstreamConfig,
    upstream_model: str,
) -> Union[CanonicalResponse, StreamingResponse]:
    instructions_extra = _extra_instructions(req, config)
    utype = (upstream_cfg.type or "openai_compat").strip().lower()
    if utype in {"openai", "grok"}:
        utype = "openai_compat"

    client = router.get_client(upstream_cfg)

    if utype == "anthropic":
        wire = build_anthropic_wire(
            req,
            upstream_model=upstream_model,
            protocol=config.features.inject_protocol,
            instructions_extra=instructions_extra,
        )
        if req.stream:
            # Responses surface over Anthropic upstream: collect non-stream for stable encode
            if req.surface == "responses":
                result = await client.messages({**wire, "stream": False}, stream=False)  # type: ignore[attr-defined]
                assert isinstance(result, dict)
                return anthropic_result_to_canonical(
                    result,
                    req=req,
                    protocol=config.features.inject_protocol,
                    strip_think=config.features.strip_think_tags,
                )
            line_iter = await client.messages(wire, stream=True)  # type: ignore[attr-defined]
            if req.fc_mode == "prompt":
                return StreamingResponse(
                    _stream_anthropic_prompt(line_iter, req=req, config=config),  # type: ignore[arg-type]
                    media_type="text/event-stream",
                    headers=_stream_headers(req.fc_mode),
                )
            return StreamingResponse(
                _stream_anthropic_native(line_iter, model=req.model),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers=_stream_headers(req.fc_mode),
            )
        result = await client.messages(wire, stream=False)  # type: ignore[attr-defined]
        assert isinstance(result, dict)
        return anthropic_result_to_canonical(
            result,
            req=req,
            protocol=config.features.inject_protocol,
            strip_think=config.features.strip_think_tags,
        )

    if utype == "gemini":
        wire = build_gemini_wire(
            req,
            protocol=config.features.inject_protocol,
            instructions_extra=instructions_extra,
        )
        if req.stream:
            line_iter = await client.generate_content(upstream_model, wire, stream=True)  # type: ignore[attr-defined]
            return StreamingResponse(
                _stream_gemini(line_iter, req=req, config=config),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers=_stream_headers(req.fc_mode),
            )
        result = await client.generate_content(upstream_model, wire, stream=False)  # type: ignore[attr-defined]
        assert isinstance(result, dict)
        return gemini_result_to_canonical(
            result,
            req=req,
            protocol=config.features.inject_protocol,
            strip_think=config.features.strip_think_tags,
        )

    # openai_compat default
    wire = build_openai_wire(
        req,
        upstream_model=upstream_model,
        protocol=config.features.inject_protocol,
        convert_developer=config.features.convert_developer_to_system,
        instructions_extra=instructions_extra,
    )
    if config.features.model_passthrough:
        wire["model"] = req.model

    if req.stream:
        line_iter = await client.chat_completions(wire, stream=True)

        if req.surface == "responses":
            generator = stream_responses_from_openai_chat(
                line_iter,  # type: ignore[arg-type]
                model=req.model,
                tools=req.tools,
                protocol=config.features.inject_protocol,
                strip_think=config.features.strip_think_tags,
                prompt_fc=(req.fc_mode == "prompt"),
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers=_stream_headers(req.fc_mode),
            )

        if req.surface == "anthropic":
            # Client asked Anthropic, upstream is OpenAI-compat — wrap text into anthropic events
            if req.fc_mode == "prompt":
                return StreamingResponse(
                    _stream_anthropic_from_openai_prompt(
                        line_iter,  # type: ignore[arg-type]
                        req=req,
                        config=config,
                    ),
                    media_type="text/event-stream",
                    headers=_stream_headers(req.fc_mode),
                )
            return StreamingResponse(
                _stream_anthropic_from_openai_native(line_iter, req=req),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers=_stream_headers(req.fc_mode),
            )

        if req.surface == "gemini":
            return StreamingResponse(
                _stream_gemini(line_iter, req=req, config=config),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers=_stream_headers(req.fc_mode),
            )

        # surface openai
        if req.fc_mode == "prompt":
            generator = stream_prompt_fc(
                line_iter,  # type: ignore[arg-type]
                model=req.model,
                tools=req.tools,
                protocol=config.features.inject_protocol,
                strip_think=config.features.strip_think_tags,
            )
        else:
            generator = stream_native_passthrough(line_iter, model=req.model)  # type: ignore[arg-type]
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers=_stream_headers(req.fc_mode),
        )

    result = await client.chat_completions(wire, stream=False)
    assert isinstance(result, dict)
    return openai_result_to_canonical(
        result,
        req=req,
        protocol=config.features.inject_protocol,
        strip_think=config.features.strip_think_tags,
    )


async def _stream_anthropic_from_openai_prompt(
    line_iter: AsyncIterator[str],
    *,
    req: CanonicalRequest,
    config: AppConfig,
) -> AsyncIterator[str]:
    """Consume OpenAI chat SSE (text), parse tools, emit Anthropic events."""
    text_parts: List[str] = []
    msg_id = f"msg_{random_id(16)}"
    yield format_sse(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        event="message_start",
    )
    try:
        async for event in iter_sse_events(line_iter):
            if event.is_done:
                break
            payload = event.json()
            if not payload:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            if delta.get("content"):
                text_parts.append(str(delta.get("content")))
    except Exception as exc:  # noqa: BLE001
        yield format_sse(
            {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            event="error",
        )
        yield format_sse({"type": "message_stop"}, event="message_stop")
        return

    full = "".join(text_parts)
    calls = parse_text_to_calls(
        full,
        req.tools,
        protocol=config.features.inject_protocol,
        strip_think=config.features.strip_think_tags,
    )
    if calls:
        for index, call in enumerate(calls):
            yield format_sse(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.id or f"toolu_{random_id(12)}",
                        "name": call.name,
                        "input": {},
                    },
                },
                event="content_block_start",
            )
            args = json.dumps(call.arguments or {}, ensure_ascii=False)
            yield format_sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                },
                event="content_block_delta",
            )
            yield format_sse({"type": "content_block_stop", "index": index}, event="content_block_stop")
        stop_reason = "tool_use"
    else:
        if full:
            yield format_sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            )
            yield format_sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": full},
                },
                event="content_block_delta",
            )
            yield format_sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
        stop_reason = "end_turn"
    yield format_sse(
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}},
        event="message_delta",
    )
    yield format_sse({"type": "message_stop"}, event="message_stop")


async def _stream_anthropic_from_openai_native(
    line_iter: AsyncIterator[str],
    *,
    req: CanonicalRequest,
) -> AsyncIterator[str]:
    """Map OpenAI native tool_calls stream into Anthropic tool_use events."""
    msg_id = f"msg_{random_id(16)}"
    yield format_sse(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        event="message_start",
    )
    text_index_started = False
    text_buf = ""
    tools: Dict[int, Dict[str, str]] = {}
    try:
        async for event in iter_sse_events(line_iter):
            if event.is_done:
                break
            payload = event.json()
            if not payload:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            if delta.get("content"):
                piece = str(delta.get("content"))
                if not text_index_started:
                    text_index_started = True
                    yield format_sse(
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                        event="content_block_start",
                    )
                text_buf += piece
                yield format_sse(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": piece},
                    },
                    event="content_block_delta",
                )
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index") or 0)
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                buf = tools.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    buf["id"] = str(tc.get("id"))
                if fn.get("name"):
                    buf["name"] = str(fn.get("name"))
                if fn.get("arguments"):
                    buf["arguments"] += str(fn.get("arguments"))
            finish = choice.get("finish_reason")
            if finish in {"tool_calls", "stop", "length"}:
                break
    except Exception as exc:  # noqa: BLE001
        yield format_sse(
            {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            event="error",
        )
        yield format_sse({"type": "message_stop"}, event="message_stop")
        return

    if text_index_started:
        yield format_sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")

    base_index = 1 if text_index_started else 0
    if tools:
        for i, idx in enumerate(sorted(tools.keys())):
            buf = tools[idx]
            block_index = base_index + i
            yield format_sse(
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": buf.get("id") or f"toolu_{random_id(12)}",
                        "name": buf.get("name") or "",
                        "input": {},
                    },
                },
                event="content_block_start",
            )
            yield format_sse(
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": buf.get("arguments") or "{}",
                    },
                },
                event="content_block_delta",
            )
            yield format_sse(
                {"type": "content_block_stop", "index": block_index},
                event="content_block_stop",
            )
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    yield format_sse(
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}},
        event="message_delta",
    )
    yield format_sse({"type": "message_stop"}, event="message_stop")


async def handle_canonical(
    req: CanonicalRequest,
    *,
    config: AppConfig,
    router: UpstreamRouter,
    fc_header: Optional[str] = None,
) -> Any:
    try:
        upstream_cfg, upstream_model = router.resolve(req.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    req.upstream_model = upstream_model
    apply_policy(req, upstream_cfg, config, header_override=fc_header)

    if req.stream and not config.features.enable_streaming:
        raise HTTPException(status_code=400, detail="streaming disabled by server config")

    name_map: Dict[str, str] = {}
    if req.fc_mode == "prompt":
        name_map = _prepare_tools_for_prompt(req, config)

    try:
        result = await _execute_once(
            req,
            config=config,
            router=router,
            upstream_cfg=upstream_cfg,
            upstream_model=upstream_model,
        )
        if isinstance(result, StreamingResponse):
            GLOBAL_METRICS.record_request(surface=req.surface, fc_mode=req.fc_mode, ok=True)
            return result

        if (
            req.fc_mode == "prompt"
            and config.features.enable_fc_error_retry
            and not result.tool_calls
            and result.content
        ):
            calls, reason = parse_with_recovery_hint(
                result.content,
                req.tools,
                protocol=config.features.inject_protocol,
                strip_think=config.features.strip_think_tags,
            )
            if reason:
                attempts = max(1, int(config.features.fc_error_retry_max))
                for _ in range(attempts):
                    GLOBAL_METRICS.record_retry()
                    retry_msg = build_retry_user_message(
                        original_output=result.content, reason=reason
                    )
                    retry_req = CanonicalRequest(
                        model=req.model,
                        messages=list(req.messages)
                        + [
                            Message(role="assistant", content=result.content),
                            Message(role="user", content=retry_msg),
                        ],
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                        stream=False,
                        surface=req.surface,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        top_p=req.top_p,
                        extra=req.extra,
                        fc_mode="prompt",
                        upstream_model=req.upstream_model,
                    )
                    retry_result = await _execute_once(
                        retry_req,
                        config=config,
                        router=router,
                        upstream_cfg=upstream_cfg,
                        upstream_model=upstream_model,
                    )
                    if isinstance(retry_result, StreamingResponse):
                        break
                    result = retry_result
                    if result.tool_calls:
                        break
                    calls, reason = parse_with_recovery_hint(
                        result.content,
                        req.tools,
                        protocol=config.features.inject_protocol,
                        strip_think=config.features.strip_think_tags,
                    )
                    if not reason:
                        break

        if name_map and result.tool_calls:
            result.tool_calls = deobfuscate_calls(result.tool_calls, name_map)

        payload = encode_client_response(req, result)
        GLOBAL_METRICS.record_request(surface=req.surface, fc_mode=req.fc_mode, ok=True)
        return JSONResponse(
            content=payload,
            headers={"X-ToolForge-FC-Mode": req.fc_mode},
        )
    except httpx.HTTPStatusError as exc:
        GLOBAL_METRICS.record_request(surface=req.surface, fc_mode=req.fc_mode, ok=False)
        status = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        GLOBAL_METRICS.record_request(surface=req.surface, fc_mode=req.fc_mode, ok=False)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
    except Exception:
        GLOBAL_METRICS.record_request(
            surface=req.surface, fc_mode=getattr(req, "fc_mode", "auto"), ok=False
        )
        raise
