/**
 * Provider context token config, last-step stamp, and intra-run snapshot.
 *
 * Focused CFG/USE coverage. Do not grow observational-memory.test.ts.
 */

import { MockLanguageModelV2, convertArrayToReadableStream } from '@internal/ai-sdk-v5/test';
import { Agent, MessageList } from '@mastra/core/agent';
import type { MastraDBMessage, MastraMessageContentV2 } from '@mastra/core/agent';
import { RequestContext } from '@mastra/core/di';
import { InMemoryDB, InMemoryMemory, InMemoryStore } from '@mastra/core/storage';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Memory } from '../../../index';
import { BufferingCoordinator } from '../buffering-coordinator';
import { ObservationalMemory } from '../observational-memory';
import { ObservationalMemoryProcessor } from '../processor';
import type { MemoryContextProvider } from '../processor';
import { normalizeProviderUsage, readLastModelUsage } from '../provider-usage';
import type { ObservationConfig } from '../types';

const noopMemoryProvider: MemoryContextProvider = {
  getContext: async () => ({
    systemMessage: undefined,
    messages: [],
    hasObservations: false,
    omRecord: null,
    continuationMessage: undefined,
    otherThreadsContext: undefined,
  }),
  persistMessages: async () => {},
};

function createTestMessage(content: string, role: 'user' | 'assistant' = 'user', id?: string): MastraDBMessage {
  const messageContent: MastraMessageContentV2 = {
    format: 2,
    parts: [{ type: 'text', text: content }],
  };

  return {
    id: id ?? `msg-${Math.random().toString(36).slice(2)}`,
    role,
    content: messageContent,
    type: 'text',
    createdAt: new Date(),
  };
}

function createMockObserverModel() {
  const observationText = `<observations>
* User discussed topic X
* Assistant explained Y
</observations>`;

  return new MockLanguageModelV2({
    doGenerate: async () => {
      throw new Error('Unexpected doGenerate call — OM should use the stream path');
    },
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        { type: 'response-metadata', id: 'mock-response', modelId: 'mock-model', timestamp: new Date() },
        { type: 'text-start', id: 'text-1' },
        { type: 'text-delta', id: 'text-1', delta: observationText },
        { type: 'text-end', id: 'text-1' },
        { type: 'finish', finishReason: 'stop', usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 } },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
      warnings: [],
    }),
  });
}

function createMockActorModel(response: string) {
  return new MockLanguageModelV2({
    doGenerate: async () => {
      throw new Error('Unexpected doGenerate call — actor should use the stream path');
    },
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        { type: 'response-metadata', id: 'a-1', modelId: 'mock-actor', timestamp: new Date() },
        { type: 'text-start', id: 'text-1' },
        { type: 'text-delta', id: 'text-1', delta: response },
        { type: 'text-end', id: 'text-1' },
        { type: 'finish', finishReason: 'stop', usage: { inputTokens: 50, outputTokens: 20, totalTokens: 70 } },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
      warnings: [],
    }),
  });
}

function createAbort() {
  return ((reason?: string) => {
    throw new Error(reason || 'Aborted');
  }) as (reason?: string) => never;
}

function createRequestContext(threadId: string, resourceId: string): RequestContext {
  const ctx = new RequestContext();
  ctx.set('MastraMemory', {
    thread: { id: threadId },
    resourceId,
  });
  ctx.set('currentDate', new Date().toISOString());
  return ctx;
}

function lastModelUsage(message: MastraDBMessage | undefined) {
  return (message?.content.metadata as { mastra?: { lastModelUsage?: unknown } } | undefined)?.mastra?.lastModelUsage;
}

describe('provider context tokens', () => {
  let storage: InMemoryMemory;
  let om: ObservationalMemory;
  let processor: ObservationalMemoryProcessor;
  const threadId = 'provider-context-tokens-thread';
  const resourceId = 'provider-context-tokens-resource';

  beforeEach(async () => {
    BufferingCoordinator.asyncBufferingOps.clear();
    BufferingCoordinator.lastBufferedBoundary.clear();
    BufferingCoordinator.observationBoundaryOwners.clear();
    BufferingCoordinator.lastBufferedAtTime.clear();
    BufferingCoordinator.reflectionBufferCycleIds.clear();

    storage = new InMemoryMemory({ db: new InMemoryDB() });
    await storage.saveThread({
      thread: {
        id: threadId,
        resourceId,
        title: 'Provider context tokens',
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {},
      },
    });

    om = new ObservationalMemory({
      storage,
      scope: 'thread',
      observation: {
        model: createMockObserverModel(),
        messageTokens: 500,
        bufferTokens: false,
        providerContextTokens: 128000,
      } as ObservationConfig,
      reflection: {
        model: createMockObserverModel(),
        observationTokens: 50000,
      },
    });
    processor = new ObservationalMemoryProcessor(om, noopMemoryProvider);

    vi.spyOn(om.observer, 'call').mockResolvedValue({
      observations: '* User discussed topic X\n* Assistant explained Y',
      usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function buildInputStepArgs(
    messageList: MessageList,
    state: Record<string, unknown>,
    stepNumber: number,
    steps: Array<{ usage?: unknown }> = [],
  ) {
    return {
      messageList,
      messages: messageList.get.all.db(),
      requestContext: createRequestContext(threadId, resourceId),
      stepNumber,
      state,
      steps,
      systemMessages: [],
      model: createMockObserverModel() as any,
      retryCount: 0,
      abort: createAbort(),
      abortSignal: new AbortController().signal,
    };
  }

  function buildOutputStepArgs(messageList: MessageList, state: Record<string, unknown>, usage: unknown) {
    return {
      messageList,
      messages: messageList.get.response.db(),
      requestContext: createRequestContext(threadId, resourceId),
      stepNumber: 0,
      state,
      steps: [],
      systemMessages: [],
      usage,
      finishReason: 'stop',
      retryCount: 0,
      abort: createAbort(),
      abortSignal: new AbortController().signal,
    };
  }

  it('stamps lastModelUsage on the last response assistant and returns the same messageList', async () => {
    const state: Record<string, unknown> = {};
    const messageList = new MessageList({ threadId, resourceId });
    messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');

    await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
    messageList.add(createTestMessage('Hi, how can I help?', 'assistant', 'msg-assistant'), 'response');

    expect(processor.processOutputStep).toEqual(expect.any(Function));
    const returned = await processor.processOutputStep!(
      buildOutputStepArgs(messageList, state, { inputTokens: 50, outputTokens: 20, totalTokens: 70 }) as any,
    );

    expect(returned).toBe(messageList);
    const lastAssistant = messageList.get.response.db().findLast(m => m.role === 'assistant');
    expect(lastModelUsage(lastAssistant)).toEqual({ inputTokens: 50, outputTokens: 20 });
  });

  it('holds a finite intra-run snapshot on the turn after the next processInputStep', async () => {
    const state: Record<string, unknown> = {};
    const messageList = new MessageList({ threadId, resourceId });
    messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');

    await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
    messageList.add(createTestMessage('Hi, how can I help?', 'assistant', 'msg-assistant'), 'response');

    if (typeof processor.processOutputStep === 'function') {
      await processor.processOutputStep(
        buildOutputStepArgs(messageList, state, { inputTokens: 50, outputTokens: 20, totalTokens: 70 }) as any,
      );
    }

    await processor.processInputStep(
      buildInputStepArgs(messageList, state, 1, [
        { usage: { inputTokens: 50, outputTokens: 20, totalTokens: 999, cachedInputTokens: 10 } },
      ]),
    );

    const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
    expect(turn.providerUsage).toEqual({ inputTokens: 50, outputTokens: 20 });
    expect(turn.providerUsageStale).toBe(false);
  });

  it('maps v4 promptTokens and completionTokens onto lastModelUsage', async () => {
    const state: Record<string, unknown> = {};
    const messageList = new MessageList({ threadId, resourceId });
    messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');
    await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
    messageList.add(createTestMessage('Hi', 'assistant', 'msg-assistant'), 'response');

    expect(processor.processOutputStep).toEqual(expect.any(Function));
    await processor.processOutputStep!(
      buildOutputStepArgs(messageList, state, { promptTokens: 50, completionTokens: 20 }) as any,
    );

    const lastAssistant = messageList.get.response.db().findLast(m => m.role === 'assistant');
    expect(lastModelUsage(lastAssistant)).toEqual({ inputTokens: 50, outputTokens: 20 });
  });

  it('does not stamp lastModelUsage when usage counts are non-finite', async () => {
    const state: Record<string, unknown> = {};
    const messageList = new MessageList({ threadId, resourceId });
    messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');
    await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
    messageList.add(createTestMessage('Hi', 'assistant', 'msg-assistant'), 'response');

    expect(processor.processOutputStep).toEqual(expect.any(Function));
    const returned = await processor.processOutputStep!(
      buildOutputStepArgs(messageList, state, { inputTokens: Number.NaN, outputTokens: 20 }) as any,
    );
    expect(returned).toBe(messageList);

    const lastAssistant = messageList.get.response.db().findLast(m => m.role === 'assistant');
    expect(lastModelUsage(lastAssistant)).toBeUndefined();
  });

  describe('normalizeProviderUsage', () => {
    it('keeps inputTokens and outputTokens and drops totalTokens and cachedInputTokens', () => {
      expect(
        normalizeProviderUsage({ inputTokens: 50, outputTokens: 20, totalTokens: 999, cachedInputTokens: 10 }),
      ).toEqual({ inputTokens: 50, outputTokens: 20 });
    });

    it('maps promptTokens and completionTokens', () => {
      expect(normalizeProviderUsage({ promptTokens: 50, completionTokens: 20 })).toEqual({
        inputTokens: 50,
        outputTokens: 20,
      });
    });

    it('returns undefined for NaN or non-finite counts', () => {
      expect(normalizeProviderUsage({ inputTokens: Number.NaN, outputTokens: 20 })).toBeUndefined();
      expect(normalizeProviderUsage({ inputTokens: 50, outputTokens: Number.POSITIVE_INFINITY })).toBeUndefined();
    });
  });

  it('stamps lastModelUsage from doStream finish after agent.stream', async () => {
    const store = new InMemoryStore();
    const observer = createMockObserverModel();
    const memory = new Memory({
      storage: store,
      options: {
        observationalMemory: {
          enabled: true,
          observation: {
            model: observer as any,
            messageTokens: 100_000,
            bufferTokens: false,
            providerContextTokens: 128000,
          } as ObservationConfig,
          reflection: {
            model: observer as any,
            observationTokens: 100_000,
          },
        },
      },
    });

    const agent = new Agent({
      id: 'provider-context-tokens-agent',
      name: 'Provider Context Tokens Agent',
      instructions: 'You are a test agent.',
      model: createMockActorModel('Hello from actor') as any,
      memory,
    });

    const streamThreadId = 'provider-tokens-stream-thread';
    const result = await agent.stream('Hello', {
      memory: { thread: streamThreadId, resource: 'provider-tokens-resource' },
    });
    await result.consumeStream();

    const memoryStore = await store.getStore('memory');
    const { messages } = await memoryStore!.listMessages({
      threadId: streamThreadId,
      orderBy: { field: 'createdAt', direction: 'ASC' },
      perPage: false,
    });
    const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
    expect(lastAssistant).toBeDefined();
    expect(readLastModelUsage(lastAssistant) ?? lastModelUsage(lastAssistant)).toEqual({
      inputTokens: 50,
      outputTokens: 20,
    });
  });
});
