import type { MastraDBMessage } from '@mastra/core/agent';

export type NormalizedProviderUsage = {
  inputTokens: number;
  outputTokens: number;
};

export function normalizeProviderUsage(usage: unknown): NormalizedProviderUsage | undefined {
  if (!usage || typeof usage !== 'object') {
    return undefined;
  }

  const raw = usage as Record<string, unknown>;
  const inputTokens = raw.inputTokens ?? raw.promptTokens;
  const outputTokens = raw.outputTokens ?? raw.completionTokens;
  if (!Number.isFinite(inputTokens) || !Number.isFinite(outputTokens)) {
    return undefined;
  }

  return { inputTokens: inputTokens as number, outputTokens: outputTokens as number };
}

export function stampLastModelUsage(message: MastraDBMessage, usage: NormalizedProviderUsage): void {
  if (!message.content.metadata) {
    message.content.metadata = {};
  }
  const metadata = message.content.metadata as { mastra?: { lastModelUsage?: NormalizedProviderUsage } };
  if (!metadata.mastra) {
    metadata.mastra = {};
  }
  metadata.mastra.lastModelUsage = { inputTokens: usage.inputTokens, outputTokens: usage.outputTokens };
}

export function readLastModelUsage(message: MastraDBMessage | undefined): NormalizedProviderUsage | undefined {
  const stamp = (message?.content?.metadata as { mastra?: { lastModelUsage?: unknown } } | undefined)?.mastra
    ?.lastModelUsage;
  return normalizeProviderUsage(stamp);
}

export function isProviderUsageStampStale(
  message: Pick<MastraDBMessage, 'createdAt'>,
  lastObservedAt?: Date | string | null,
): boolean {
  if (lastObservedAt == null || lastObservedAt === '') {
    return false;
  }
  const cursor = new Date(lastObservedAt).getTime();
  if (!Number.isFinite(cursor)) {
    return false;
  }
  const created = new Date(message.createdAt).getTime();
  if (!Number.isFinite(created)) {
    return true;
  }
  return created <= cursor;
}
