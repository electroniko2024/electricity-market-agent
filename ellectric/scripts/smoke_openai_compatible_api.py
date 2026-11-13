"""Verify tool calling and JSON-mode output from an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

try:
    import ellectric.llm  # noqa: F401  (loads the project-root .env)
except ImportError:
    pass


DEFAULT_BASE_URL = "https://aispot.swj0227.icu/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "confirm_trade_date",
            "description": "Return the requested trade date for this API smoke test.",
            "parameters": {
                "type": "object",
                "properties": {"trade_date": {"type": "string", "format": "date"}},
                "required": ["trade_date"],
                "additionalProperties": False,
            },
        },
    }
]


class EndpointError(RuntimeError):
    pass


def _post_chat_completion(
    *, base_url: str, model: str, api_key: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps({"model": model, **payload}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise EndpointError(f"API 请求失败，HTTP {error.code}") from None
    except urllib.error.URLError as error:
        reason = error.reason
        reason_name = type(reason).__name__
        raise EndpointError(f"API 连接失败，原因类型 {reason_name}") from None
    except TimeoutError:
        raise EndpointError("API 请求超时") from None

    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EndpointError("API 返回内容不是有效 JSON") from None
    if not isinstance(result, dict):
        raise EndpointError("API 返回结构不是 JSON 对象")
    return result


def _first_message(result: dict[str, Any]) -> dict[str, Any]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise EndpointError("API 响应缺少 choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise EndpointError("API 响应缺少 assistant message")
    return message


def run_smoke_test(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float = 45.0,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("未配置 API key；请在 WSL 设置 ELLECTRIC_LLM_API_KEY")
    tool_result = _post_chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call confirm_trade_date exactly once with trade_date "
                        "2025-10-01. Do not answer without the tool."
                    ),
                }
            ],
            "tools": TOOL_SCHEMA,
            "tool_choice": "required",
        },
    )
    tool_message = _first_message(tool_result)
    tool_calls = tool_message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise EndpointError("工具调用验收失败：预期恰好一次 tool call")
    function_call = tool_calls[0].get("function", {})
    if function_call.get("name") != "confirm_trade_date":
        raise EndpointError("工具调用验收失败：函数名称不匹配")
    try:
        arguments = json.loads(function_call.get("arguments", ""))
    except json.JSONDecodeError:
        raise EndpointError("工具调用验收失败：arguments 不是有效 JSON") from None
    if arguments != {"trade_date": "2025-10-01"}:
        raise EndpointError("工具调用验收失败：函数参数不匹配")

    structured_result = _post_chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return JSON with exactly these fields: status set to 'ok', "
                        "trade_date set to '2025-10-01', sample_count set to 96."
                    ),
                }
            ],
            "response_format": {"type": "json_object"},
        },
    )
    structured_content = _first_message(structured_result).get("content")
    try:
        structured = json.loads(structured_content)
    except (json.JSONDecodeError, TypeError):
        raise EndpointError("结构化输出验收失败：content 不是有效 JSON") from None
    if structured != {
        "status": "ok",
        "trade_date": "2025-10-01",
        "sample_count": 96,
    }:
        raise EndpointError("结构化输出验收失败：返回字段或值不匹配")

    return {
        "base_url": base_url,
        "model": model,
        "tool_call": "passed",
        "json_object": "passed",
        "api_key_printed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ELLECTRIC_LLM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--model", default=os.environ.get("ELLECTRIC_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    api_key = os.environ.get("ELLECTRIC_LLM_API_KEY", "")
    result = run_smoke_test(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EndpointError, ValueError) as error:
        raise SystemExit(str(error)) from None
