export interface AnalysisCallbacks {
  onEvent?: (event: Record<string, unknown>) => void;
  onError?: (message: string) => void;
}

export async function streamAnalysis(
  request: { question: string; as_of: string; mode: "historical_analysis" | "historical_forecast" },
  callbacks: AnalysisCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/analyze/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) {
    callbacks.onError?.(`HTTP ${response.status}`);
    return;
  }
  if (!response.body) {
    callbacks.onError?.("分析服务没有返回事件流");
    return;
  }
  try {
    for await (const event of readSSE(response.body)) {
      callbacks.onEvent?.(event);
      if (event.type === "error") {
        callbacks.onError?.(String(event.message ?? "分析失败"));
      }
    }
  } catch (error) {
    if (!signal?.aborted) callbacks.onError?.((error as Error).message);
  }
}

export async function createConversation(): Promise<string> {
  const response = await fetch("/conversations", { method: "POST" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).conversation_id as string;
}

export async function streamTurn(
  conversationId: string,
  request: { message: string; as_of: string; mode: "historical_analysis" | "historical_forecast" },
  callbacks: AnalysisCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/conversations/${conversationId}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || !response.body) {
    callbacks.onError?.(`HTTP ${response.status}`);
    return;
  }
  try {
    for await (const event of readSSE(response.body)) {
      callbacks.onEvent?.(event);
      if (event.type === "error") callbacks.onError?.(String(event.message ?? "失败"));
    }
  } catch (error) {
    if (!signal?.aborted) callbacks.onError?.((error as Error).message);
  }
}

async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<Record<string, unknown>> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        for (const line of part.split("\n")) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6).trim();
            if (raw && raw !== "[DONE]") yield JSON.parse(raw);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
