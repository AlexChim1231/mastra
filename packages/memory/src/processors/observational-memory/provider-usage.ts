import type { MastraDBMessage } from '@mastra/core/agent';

export type NormalizedProviderUsage = {
  inputTokens: number;
  outputTokens: number;
};

export function normalizeProviderUsage(_usage: unknown): NormalizedProviderUsage | undefined {
  return { inputTokens: -1, outputTokens: -1 };
}

export function stampLastModelUsage(_message: MastraDBMessage, _usage: NormalizedProviderUsage): void {}

export function readLastModelUsage(_message: MastraDBMessage | undefined): NormalizedProviderUsage | undefined {
  return undefined;
}
