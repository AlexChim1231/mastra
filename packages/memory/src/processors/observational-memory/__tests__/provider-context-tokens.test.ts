/**
 * Provider context token config, last-step stamp, and intra-run snapshot.
 *
 * Focused CFG/USE coverage. Do not grow observational-memory.test.ts.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MockLanguageModelV2, convertArrayToReadableStream } from '@internal/ai-sdk-v5/test';
import { Agent, MessageList } from '@mastra/core/agent';
import type { MastraDBMessage, MastraMessageContentV2 } from '@mastra/core/agent';
import { RequestContext } from '@mastra/core/di';
import { InMemoryDB, InMemoryMemory, InMemoryStore } from '@mastra/core/storage';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Memory } from '../../../index';
import { BufferingCoordinator } from '../buffering-coordinator';
import { OBSERVATIONAL_MEMORY_DEFAULTS } from '../constants';
import { ObservationalMemory } from '../observational-memory';
import { ObservationalMemoryProcessor } from '../processor';
import type { MemoryContextProvider } from '../processor';
import { normalizeProviderUsage, readLastModelUsage, stampLastModelUsage } from '../provider-usage';

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

function createStampedAssistant(opts: {
  id: string;
  inputTokens: number;
  outputTokens: number;
  createdAt?: Date;
}): MastraDBMessage {
  const message = createTestMessage('prior assistant', 'assistant', opts.id);
  if (opts.createdAt) {
    message.createdAt = opts.createdAt;
  }
  stampLastModelUsage(message, { inputTokens: opts.inputTokens, outputTokens: opts.outputTokens });
  return message;
}

function rememberedProvider(messages: MastraDBMessage[]): MemoryContextProvider {
  return {
    getContext: async () => ({
      systemMessage: undefined,
      messages,
      hasObservations: false,
      omRecord: null,
      continuationMessage: undefined,
      otherThreadsContext: undefined,
    }),
    persistMessages: async () => {},
  };
}

/** Match the agent loop: append a trailing step-start before processInputStep. */
function openLoopStepBoundary(messageList: MessageList) {
  expect(messageList.openStepBoundary().appended).toBe(true);
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
      },
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
        { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } },
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
          },
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

  describe('providerContextTokens config', () => {
    function createOm(providerContextTokens?: number) {
      return new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500,
          bufferTokens: false,
          ...(providerContextTokens !== undefined ? { providerContextTokens } : {}),
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
    }

    it.each([0, Number.NaN, Number.POSITIVE_INFINITY, -1])('rejects invalid providerContextTokens %s', value => {
      expect(() => createOm(value)).toThrow(/observation\.providerContextTokens must be a finite number > 0, got /);
      expect(() => createOm(value)).toThrow(String(value));
    });

    it('accepts a non-integer finite number greater than 0', () => {
      expect(() => createOm(1.5)).not.toThrow();
    });

    it('accepts 128000', () => {
      expect(() => createOm(128000)).not.toThrow();
    });

    it('constructs when providerContextTokens is unset', () => {
      expect(() => createOm()).not.toThrow();
    });

    it('does not default providerContextTokens in observation defaults', () => {
      expect('providerContextTokens' in OBSERVATIONAL_MEMORY_DEFAULTS.observation).toBe(false);
    });

    it('copies providerContextTokens by name in _initOMEngine', () => {
      const source = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../../../index.ts'), 'utf8');
      expect(source).toMatch(/providerContextTokens:\s*omConfig\.observation\.providerContextTokens/);
    });
  });

  describe('unset providerContextTokens', () => {
    let unsetProcessor: ObservationalMemoryProcessor;

    beforeEach(() => {
      const unsetOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500,
          bufferTokens: false,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      unsetProcessor = new ObservationalMemoryProcessor(unsetOm, noopMemoryProvider);
    });

    it('does not expose processOutputStep on the processor instance', () => {
      expect('processOutputStep' in unsetProcessor).toBe(false);
    });

    it('processInputStep still returns the same messageList', async () => {
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');
      const returned = await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));
      expect(returned).toBe(messageList);
    });

    it('does not mark providerUsageStale after observe when providerContextTokens is unset', async () => {
      const observeOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 1,
          bufferTokens: false,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      vi.spyOn(observeOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const observeProcessor = new ObservationalMemoryProcessor(observeOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('please summarise this. '.repeat(40), 'user', 'msg-user-unset-obs'), 'input');

      await observeProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));

      expect(observeOm.observer.call).toHaveBeenCalled();
      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toBeUndefined();
      expect(turn.providerUsageStale).not.toBe(true);
    });

    it('does not mark providerUsageStale after step-0 activation when providerContextTokens is unset', async () => {
      const activateOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500,
          bufferTokens: false,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      const origActivate = activateOm.activate.bind(activateOm);
      vi.spyOn(activateOm, 'activate').mockImplementation(async opts => {
        const result = await origActivate(opts);
        return { ...result, activated: true, activatedMessageIds: result.activatedMessageIds ?? [] };
      });
      const activateProcessor = new ObservationalMemoryProcessor(activateOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await activateProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));

      expect(activateOm.activate).toHaveBeenCalled();
      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toBeUndefined();
      expect(turn.providerUsageStale).not.toBe(true);
    });
  });

  it('does not stamp lastModelUsage when usage counts are missing', async () => {
    const state: Record<string, unknown> = {};
    const messageList = new MessageList({ threadId, resourceId });
    messageList.add(createTestMessage('Hello there', 'user', 'msg-user'), 'input');
    await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
    messageList.add(createTestMessage('Hi', 'assistant', 'msg-assistant'), 'response');

    expect(processor.processOutputStep).toEqual(expect.any(Function));
    const returned = await processor.processOutputStep!(
      buildOutputStepArgs(messageList, state, { outputTokens: 20 }) as any,
    );
    expect(returned).toBe(messageList);

    const lastAssistant = messageList.get.response.db().findLast(m => m.role === 'assistant');
    expect(lastModelUsage(lastAssistant)).toBeUndefined();
  });

  it('preserves existing content.metadata keys and other mastra keys when stamping', () => {
    const message = createTestMessage('hi', 'assistant', 'asst-meta');
    message.content.metadata = { source: 'ibox', mastra: { sealed: true } };
    stampLastModelUsage(message, { inputTokens: 8, outputTokens: 3 });
    expect(message.content.metadata).toEqual({
      source: 'ibox',
      mastra: { sealed: true, lastModelUsage: { inputTokens: 8, outputTokens: 3 } },
    });
  });

  describe('step-0 remembered stamp', () => {
    it('sets turn.providerUsage from the last remembered assistant stamp', async () => {
      const stamped = createStampedAssistant({ id: 'asst-latest', inputTokens: 80, outputTokens: 30 });
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toEqual({ inputTokens: 80, outputTokens: 30 });
      expect(turn.providerUsageStale).toBe(false);
    });

    it('does not fall back to an older stamped assistant when the last has no stamp', async () => {
      const older = createStampedAssistant({ id: 'asst-old', inputTokens: 80, outputTokens: 30 });
      const latest = createTestMessage('latest unstamped', 'assistant', 'asst-new');
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([older, latest]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown };
      expect(turn.providerUsage).toBeUndefined();
    });

    it('treats a stamp as missing when assistant createdAt is at or before lastObservedAt', async () => {
      const createdAt = new Date('2026-01-01T00:00:00.000Z');
      const lastObservedAt = new Date('2026-01-01T00:00:00.000Z');
      const stamped = createStampedAssistant({
        id: 'asst-stale-cursor',
        inputTokens: 80,
        outputTokens: 30,
        createdAt,
      });
      const orig = om.getOrCreateRecord.bind(om);
      vi.spyOn(om, 'getOrCreateRecord').mockImplementation(async (tid, rid) => {
        const rec = await orig(tid, rid);
        rec.lastObservedAt = lastObservedAt;
        return rec;
      });
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown };
      expect(turn.providerUsage).toBeUndefined();
    });

    it('installs a stamp when createdAt is after lastObservedAt and the assistant is unobserved', async () => {
      const createdAt = new Date('2026-01-02T00:00:00.000Z');
      const lastObservedAt = new Date('2026-01-01T00:00:00.000Z');
      const stamped = createStampedAssistant({
        id: 'asst-fresh-unobserved',
        inputTokens: 80,
        outputTokens: 30,
        createdAt,
      });
      const orig = om.getOrCreateRecord.bind(om);
      vi.spyOn(om, 'getOrCreateRecord').mockImplementation(async (tid, rid) => {
        const rec = await orig(tid, rid);
        rec.lastObservedAt = lastObservedAt;
        rec.observedMessageIds = [];
        return rec;
      });
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toEqual({ inputTokens: 80, outputTokens: 30 });
      expect(turn.providerUsageStale).toBe(false);
    });

    it('skips a stamp when createdAt is before lastObservedAt', async () => {
      const createdAt = new Date('2025-12-31T00:00:00.000Z');
      const lastObservedAt = new Date('2026-01-01T00:00:00.000Z');
      const stamped = createStampedAssistant({
        id: 'asst-before-cursor',
        inputTokens: 80,
        outputTokens: 30,
        createdAt,
      });
      const orig = om.getOrCreateRecord.bind(om);
      vi.spyOn(om, 'getOrCreateRecord').mockImplementation(async (tid, rid) => {
        const rec = await orig(tid, rid);
        rec.lastObservedAt = lastObservedAt;
        return rec;
      });
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown };
      expect(turn.providerUsage).toBeUndefined();
    });

    it('does not install a stamp when the assistant is already in observedMessageIds', async () => {
      const createdAt = new Date('2026-01-02T00:00:00.000Z');
      const lastObservedAt = new Date('2026-01-01T00:00:00.000Z');
      const stamped = createStampedAssistant({
        id: 'asst-already-observed',
        inputTokens: 80,
        outputTokens: 30,
        createdAt,
      });
      const orig = om.getOrCreateRecord.bind(om);
      vi.spyOn(om, 'getOrCreateRecord').mockImplementation(async (tid, rid) => {
        const rec = await orig(tid, rid);
        rec.lastObservedAt = lastObservedAt;
        rec.observedMessageIds = [stamped.id];
        return rec;
      });
      const step0Processor = new ObservationalMemoryProcessor(om, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await step0Processor.processInputStep(buildInputStepArgs(messageList, state, 0));

      const turn = state.__omTurn as { providerUsage?: unknown };
      expect(turn.providerUsage).toBeUndefined();
    });
  });

  describe('usage snapshot gating', () => {
    it('does not install a snapshot when providerContextTokens is unset', async () => {
      const unsetOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500,
          bufferTokens: false,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      const unsetProcessor = new ObservationalMemoryProcessor(unsetOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('Hello there', 'user', 'msg-user-unset'), 'input');

      await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));
      await unsetProcessor.processInputStep(
        buildInputStepArgs(messageList, state, 1, [{ usage: { inputTokens: 50, outputTokens: 20 } }]),
      );

      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toBeUndefined();
    });

    it('does not clear providerUsageStale when later usage is invalid', async () => {
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('Hello there', 'user', 'msg-user-stale'), 'input');

      await processor.processInputStep(buildInputStepArgs(messageList, state, 0));
      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale: boolean };
      turn.providerUsageStale = true;

      await processor.processInputStep(
        buildInputStepArgs(messageList, state, 1, [{ usage: { inputTokens: Number.NaN, outputTokens: 20 } }]),
      );

      expect(turn.providerUsageStale).toBe(true);
      expect(turn.providerUsage).toBeUndefined();
    });
  });

  describe('stale snapshot after observe or activation', () => {
    function createCaptureOm(observation: { messageTokens: number; bufferTokens?: number | false }) {
      return new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          bufferTokens: false,
          providerContextTokens: 128000,
          ...observation,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
    }

    it('clears providerUsage and marks stale after a successful observe', async () => {
      const observeOm = createCaptureOm({ messageTokens: 1 });
      vi.spyOn(observeOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const stamped = createStampedAssistant({ id: 'asst-pre-observe', inputTokens: 80, outputTokens: 30 });
      const observeProcessor = new ObservationalMemoryProcessor(observeOm, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('please summarise this. '.repeat(40), 'user', 'msg-user-obs'), 'input');

      await observeProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));

      expect(observeOm.observer.call).toHaveBeenCalled();
      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toBeUndefined();
      expect(turn.providerUsageStale).toBe(true);
    });

    it('clears providerUsage and marks stale after step-0 activation', async () => {
      const activateOm = createCaptureOm({ messageTokens: 500 });
      vi.spyOn(activateOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const origActivate = activateOm.activate.bind(activateOm);
      vi.spyOn(activateOm, 'activate').mockImplementation(async opts => {
        const result = await origActivate(opts);
        return { ...result, activated: true, activatedMessageIds: result.activatedMessageIds ?? [] };
      });
      const stamped = createStampedAssistant({ id: 'asst-pre-activate', inputTokens: 80, outputTokens: 30 });
      const activateProcessor = new ObservationalMemoryProcessor(activateOm, rememberedProvider([stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await activateProcessor.processInputStep(buildInputStepArgs(messageList, state, 0));

      expect(activateOm.activate).toHaveBeenCalled();
      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsage).toBeUndefined();
      expect(turn.providerUsageStale).toBe(true);

      messageList.add(createTestMessage('later actor reply', 'assistant', 'msg-asst-later'), 'response');
      expect(activateProcessor.processOutputStep).toEqual(expect.any(Function));
      await activateProcessor.processOutputStep!(
        buildOutputStepArgs(messageList, state, { inputTokens: 11, outputTokens: 7 }) as any,
      );
      await activateProcessor.processInputStep(
        buildInputStepArgs(messageList, state, 1, [{ usage: { inputTokens: 11, outputTokens: 7 } }]),
      );

      expect(turn.providerUsage).toEqual({ inputTokens: 11, outputTokens: 7 });
      expect(turn.providerUsageStale).toBe(false);
    });
  });

  describe('within-run provider occupancy trigger', () => {
    const providerCap = 40000;
    const lastStepUsage = { inputTokens: 39980, outputTokens: 10 };

    function createProviderOnlyOm() {
      return new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500000,
          bufferTokens: false,
          providerContextTokens: providerCap,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
    }

    it('observes on step 1 when tool-result added-since crosses the cap while last-step usage stays under it', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-a1'), 'memory');
      messageList.add(createTestMessage('short memory three', 'user', 'mem-u2'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      const slack = providerCap - (lastStepUsage.inputTokens + lastStepUsage.outputTokens);
      expect(lastStepUsage.inputTokens + lastStepUsage.outputTokens).toBeLessThan(providerCap);

      const resultPart = {
        type: 'tool-invocation' as const,
        toolInvocation: {
          state: 'result' as const,
          toolCallId: 'call-added-since',
          toolName: 'lookup',
          args: {},
          result: 'tool result body that must exceed remaining slack. '.repeat(8),
        },
      };
      const addedSinceTokens = triggerOm.getTokenCounter().countMessage({
        id: 'synthetic-added-since',
        role: 'assistant',
        type: 'text',
        createdAt: new Date(),
        content: { format: 2, parts: [resultPart] },
      });
      expect(addedSinceTokens).toBeGreaterThan(slack);

      messageList.add(
        {
          id: 'asst-mid-loop',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: {
            format: 2,
            parts: [{ type: 'text', text: 'I will look that up after the last model call. '.repeat(20) }, resultPart],
          },
        },
        'response',
      );
      openLoopStepBoundary(messageList);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: lastStepUsage }]));

      expect(triggerOm.observer.call).toHaveBeenCalled();
    });

    function countSyntheticParts(om: ObservationalMemory, parts: MastraDBMessage['content']['parts']) {
      return om.getTokenCounter().countMessage({
        id: 'om-provider-added-since',
        role: 'assistant',
        type: 'text',
        createdAt: new Date(),
        content: { format: 2, parts },
      });
    }

    it('does not observe when one tool-result copy stays under the cap but two copies would cross it', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-dup-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-dup-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      const resultPart = {
        type: 'tool-invocation' as const,
        toolInvocation: {
          state: 'result' as const,
          toolCallId: 'call-once',
          toolName: 'lookup',
          args: {},
          result: 'tool result body sized so one copy stays under slack. '.repeat(6),
        },
      };
      const onceTokens = countSyntheticParts(triggerOm, [resultPart]);
      const twiceTokens = countSyntheticParts(triggerOm, [resultPart, resultPart]);
      expect(onceTokens).toBeGreaterThan(0);
      expect(twiceTokens).toBeGreaterThan(onceTokens);

      const underCapUsage = {
        inputTokens: providerCap - onceTokens - 1,
        outputTokens: 0,
      };
      expect(underCapUsage.inputTokens + underCapUsage.outputTokens + onceTokens).toBeLessThan(providerCap);
      expect(underCapUsage.inputTokens + underCapUsage.outputTokens + twiceTokens).toBeGreaterThanOrEqual(providerCap);
      expect(underCapUsage.inputTokens + underCapUsage.outputTokens).toBeLessThan(providerCap);

      messageList.add(
        {
          id: 'asst-dup-result',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: {
            format: 2,
            parts: [{ type: 'text', text: 'I will look that up after the last model call. '.repeat(20) }, resultPart],
          },
        },
        'response',
      );
      openLoopStepBoundary(messageList);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: underCapUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe from a long assistant text part without a tool result', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-text-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-text-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      const slack = providerCap - (lastStepUsage.inputTokens + lastStepUsage.outputTokens);
      const textPart = {
        type: 'text' as const,
        text: 'I will look that up after the last model call. '.repeat(200),
      };
      expect(countSyntheticParts(triggerOm, [textPart])).toBeGreaterThan(slack);

      messageList.add(
        {
          id: 'asst-long-text-only',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: { format: 2, parts: [textPart] },
        },
        'response',
      );

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: lastStepUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    function crossingToolResultPart() {
      return {
        type: 'tool-invocation' as const,
        toolInvocation: {
          state: 'result' as const,
          toolCallId: 'call-added-since',
          toolName: 'lookup',
          args: {},
          result: 'tool result body that must exceed remaining slack. '.repeat(8),
        },
      };
    }

    function addCrossingAssistant(
      messageList: MessageList,
      id: string,
      extraParts: MastraDBMessage['content']['parts'] = [],
      openBoundary = true,
    ) {
      messageList.add(
        {
          id,
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: {
            format: 2,
            parts: [
              { type: 'text', text: 'I will look that up after the last model call. '.repeat(20) },
              crossingToolResultPart(),
              ...extraParts,
            ],
          },
        },
        'response',
      );
      if (openBoundary) {
        openLoopStepBoundary(messageList);
      }
    }

    it('does not observe when last-step usage is missing at high messageTokens', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-miss-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-miss-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      addCrossingAssistant(messageList, 'asst-missing-usage');
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, []));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe when last-step usage is non-finite at high messageTokens', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-nan-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-nan-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      addCrossingAssistant(messageList, 'asst-nan-usage');
      await triggerProcessor.processInputStep(
        buildInputStepArgs(messageList, state, 1, [{ usage: { inputTokens: Number.NaN, outputTokens: 10 } }]),
      );

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe when the usage snapshot is stale at high messageTokens', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-stale-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-stale-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      addCrossingAssistant(messageList, 'asst-stale-first');
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: lastStepUsage }]));
      expect(triggerOm.observer.call).toHaveBeenCalledTimes(1);

      const turn = state.__omTurn as { providerUsage?: unknown; providerUsageStale?: boolean };
      expect(turn.providerUsageStale).toBe(true);

      addCrossingAssistant(messageList, 'asst-stale-second');
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 2, []));

      expect(triggerOm.observer.call).toHaveBeenCalledTimes(1);
    });

    it('still observes from the local messageTokens check when usage is missing', async () => {
      const localOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 1,
          bufferTokens: false,
          providerContextTokens: providerCap,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      vi.spyOn(localOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const localProcessor = new ObservationalMemoryProcessor(localOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('please summarise this. '.repeat(40), 'user', 'msg-local-missing'), 'input');

      await localProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));

      expect(localOm.observer.call).toHaveBeenCalled();
    });

    it('observes once when local shouldObserve and the provider trigger are both true', async () => {
      const bothOm = new ObservationalMemory({
        storage,
        scope: 'thread',
        observation: {
          model: createMockObserverModel(),
          messageTokens: 200,
          bufferTokens: false,
          providerContextTokens: providerCap,
        },
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
      vi.spyOn(bothOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const bothProcessor = new ObservationalMemoryProcessor(bothOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'msg-both-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'msg-both-a1'), 'memory');

      await bothProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(bothOm.observer.call).not.toHaveBeenCalled();

      addCrossingAssistant(messageList, 'asst-both-triggers');
      await bothProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: lastStepUsage }]));

      expect(bothOm.observer.call).toHaveBeenCalledTimes(1);
    });

    it('does not observe when an incomplete tool call is pending even if the estimate crosses the cap', async () => {
      const triggerOm = createProviderOnlyOm();
      vi.spyOn(triggerOm.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'mem-inc-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'mem-inc-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      addCrossingAssistant(
        messageList,
        'asst-incomplete-tool',
        [
          {
            type: 'tool-invocation',
            toolInvocation: {
              state: 'call',
              toolCallId: 'call-pending',
              toolName: 'lookup',
              args: {},
            },
          },
        ],
        false,
      );
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: lastStepUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });
  });

  describe('provider occupancy trigger', () => {
    const providerCap = 40000;
    const overCapUsage = { inputTokens: 80000, outputTokens: 200 };
    const belowCapUsage = { inputTokens: 39980, outputTokens: 10 };

    function mockObserverCall(target: ObservationalMemory) {
      return vi.spyOn(target.observer, 'call').mockResolvedValue({
        observations: '* User discussed topic X\n* Assistant explained Y',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
    }

    function mockObserverMultiThread(target: ObservationalMemory, observeThreadId: string) {
      return vi.spyOn(target.observer, 'callMultiThread').mockResolvedValue({
        results: new Map([[observeThreadId, { observations: '* User discussed topic X' }]]),
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      });
    }

    function createOccupancyOm(observation: Record<string, unknown> = {}, scope: 'thread' | 'resource' = 'thread') {
      return new ObservationalMemory({
        storage,
        scope,
        observation: {
          model: createMockObserverModel(),
          messageTokens: 500000,
          bufferTokens: false,
          ...observation,
        } as ConstructorParameters<typeof ObservationalMemory>[0]['observation'],
        reflection: {
          model: createMockObserverModel(),
          observationTokens: 50000,
        },
      });
    }

    function countSyntheticParts(target: ObservationalMemory, parts: MastraDBMessage['content']['parts']) {
      return target.getTokenCounter().countMessage({
        id: 'om-provider-added-since',
        role: 'assistant',
        type: 'text',
        createdAt: new Date(),
        content: { format: 2, parts },
      });
    }

    function sumCountMessage(target: ObservationalMemory, messages: MastraDBMessage[]) {
      return messages.reduce((sum, message) => sum + target.getTokenCounter().countMessage(message), 0);
    }

    function createBulkMessages(count: number, bulkThreadId: string): MastraDBMessage[] {
      const base = Date.now() - count * 1000;
      return Array.from({ length: count }, (_, i) => ({
        ...createTestMessage(
          `Message ${i}: `.padEnd(200, 'x'),
          i % 2 === 0 ? 'user' : 'assistant',
          `${bulkThreadId}-msg-${i}`,
        ),
        threadId: bulkThreadId,
        resourceId,
        createdAt: new Date(base + i * 1000),
      }));
    }

    function tinyToolResultPart() {
      return {
        type: 'tool-invocation' as const,
        toolInvocation: {
          state: 'result' as const,
          toolCallId: 'call-tiny',
          toolName: 'lookup',
          args: {},
          result: 'ok',
        },
      };
    }

    function createAttachmentUnderweighMessage(): MastraDBMessage {
      return {
        id: 'qa02-attachments',
        role: 'user',
        type: 'text',
        createdAt: new Date(),
        content: {
          format: 2,
          parts: [
            { type: 'text', text: 'please inspect these' },
            { type: 'image', image: 'https://example.com/x.jpg' },
            { type: 'file', mimeType: 'application/pdf', data: 'https://example.com/a.pdf' },
          ],
        },
      };
    }

    function crossingToolResultPart() {
      return {
        type: 'tool-invocation' as const,
        toolInvocation: {
          state: 'result' as const,
          toolCallId: 'call-added-since',
          toolName: 'lookup',
          args: {},
          result: 'tool result body that must exceed remaining slack. '.repeat(8),
        },
      };
    }

    it('does not observe when providerContextTokens is unset even if last-step usage is large', async () => {
      const unsetOm = createOccupancyOm({
        messageTokens: 500000,
        bufferTokens: false,
      });
      mockObserverCall(unsetOm);
      const unsetProcessor = new ObservationalMemoryProcessor(unsetOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short one', 'user', 'unset-cap-u1'), 'memory');
      messageList.add(createTestMessage('short two', 'assistant', 'unset-cap-a1'), 'memory');

      await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(unsetOm.observer.call).not.toHaveBeenCalled();
    });

    it('observes at step 0 when a new input.db() message crosses the cap while stamped usage stays under it', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const olderHistory = createTestMessage(
        'remembered history that must stay in the remembered bucket. '.repeat(40),
        'user',
        'hist-old-user',
      );
      const stamped = createStampedAssistant({
        id: 'asst-cross-turn',
        inputTokens: belowCapUsage.inputTokens,
        outputTokens: belowCapUsage.outputTokens,
      });
      expect(belowCapUsage.inputTokens + belowCapUsage.outputTokens).toBeLessThan(providerCap);
      const slack = providerCap - (belowCapUsage.inputTokens + belowCapUsage.outputTokens);
      const newUser = createTestMessage(
        'fresh user text that must exceed remaining slack. '.repeat(8),
        'user',
        'input-db-cross',
      );
      expect(triggerOm.getTokenCounter().countMessage(newUser)).toBeGreaterThan(slack);
      expect(triggerOm.getTokenCounter().countMessage(olderHistory)).toBeGreaterThan(slack);

      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, rememberedProvider([olderHistory, stamped]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(newUser, 'input');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));

      expect(triggerOm.observer.call).toHaveBeenCalled();
    });

    it('does not observe when tiny added-since stays under slack even if text, history, or countMessages would cross', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const longHistory = createTestMessage(
        'long remembered history that would exceed remaining slack. '.repeat(40),
        'user',
        'hist-paired-neg',
      );
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, rememberedProvider([longHistory]));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      const tinyUser = createTestMessage('hi', 'user', 'tiny-input-db');
      const resultPart = tinyToolResultPart();
      const tinyResultTokens = countSyntheticParts(triggerOm, [resultPart]);
      const tinyUserTokens = triggerOm.getTokenCounter().countMessage(tinyUser);
      const added = tinyUserTokens + tinyResultTokens;
      expect(added).toBeGreaterThan(0);
      expect(added).toBeLessThan(24);

      const slack = Math.max(added + 1, 8);
      expect(slack).toBeLessThanOrEqual(24);
      const usage = { inputTokens: providerCap - slack, outputTokens: 0 };
      expect(usage.inputTokens + usage.outputTokens).toBeLessThan(providerCap);
      expect(usage.inputTokens + usage.outputTokens + added).toBeLessThan(providerCap);
      expect(
        usage.inputTokens + usage.outputTokens + triggerOm.getTokenCounter().countMessages([tinyUser]),
      ).toBeGreaterThanOrEqual(providerCap);

      const longTextPart = {
        type: 'text' as const,
        text: 'I will look that up after the last model call. '.repeat(200),
      };
      expect(countSyntheticParts(triggerOm, [longTextPart])).toBeGreaterThan(slack);
      expect(triggerOm.getTokenCounter().countMessage(longHistory)).toBeGreaterThan(slack);

      messageList.add(tinyUser, 'input');
      messageList.add(
        {
          id: 'asst-paired-neg',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: { format: 2, parts: [longTextPart, resultPart] },
        },
        'response',
      );
      openLoopStepBoundary(messageList);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe on step 1 when leftover step-0 input would cross the cap if recounted', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      const leftover = createTestMessage(
        'fresh user text that must exceed remaining slack. '.repeat(8),
        'user',
        'leftover-step0-input',
      );
      const leftoverTokens = triggerOm.getTokenCounter().countMessage(leftover);
      expect(leftoverTokens).toBeGreaterThan(0);
      const usage = { inputTokens: providerCap - leftoverTokens + 1, outputTokens: 0 };
      expect(usage.inputTokens + usage.outputTokens).toBeLessThan(providerCap);
      expect(usage.inputTokens + usage.outputTokens + leftoverTokens).toBeGreaterThanOrEqual(providerCap);

      messageList.add(leftover, 'input');
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('observes on step 1 when new mid-loop input crosses the cap', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      const leftover = createTestMessage('step-0 leftover prompt', 'user', 'leftover-kept-small');
      const leftoverTokens = triggerOm.getTokenCounter().countMessage(leftover);
      expect(leftoverTokens).toBeGreaterThan(0);
      messageList.add(leftover, 'input');
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();

      const slack = leftoverTokens + 8;
      const usage = { inputTokens: providerCap - slack, outputTokens: 0 };
      expect(usage.inputTokens + usage.outputTokens).toBeLessThan(providerCap);
      expect(usage.inputTokens + usage.outputTokens + leftoverTokens).toBeLessThan(providerCap);

      const newInput = createTestMessage(
        'genuinely new mid-loop user text that must exceed remaining slack. '.repeat(8),
        'user',
        'new-mid-loop-input',
      );
      const newTokens = triggerOm.getTokenCounter().countMessage(newInput);
      expect(newTokens).toBeGreaterThan(slack);
      messageList.add(newInput, 'input');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage }]));

      expect(triggerOm.observer.call).toHaveBeenCalled();
    });

    it('does not observe overhead-only occupancy when the unobserved window is empty at floor 0', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('observes overhead-only occupancy when unobserved local tokens are above floor 0', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      const unobserved = createTestMessage('keep one unobserved token', 'user', 'floor0-unobserved');
      expect(triggerOm.getTokenCounter().countMessage(unobserved)).toBeGreaterThan(0);
      messageList.add(unobserved, 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(triggerOm.observer.call).toHaveBeenCalled();
    });

    it('does not observe when unobserved local tokens are at or below a positive retention floor', async () => {
      const triggerOm = createOccupancyOm({
        messageTokens: 30000,
        bufferTokens: 5000,
        bufferActivation: 0.7,
        providerContextTokens: providerCap,
      });
      mockObserverCall(triggerOm);
      const record = await triggerOm.getOrCreateRecord(threadId, resourceId);
      vi.spyOn(triggerOm, 'buffer').mockResolvedValue({ buffered: false, record });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      const unobserved = [createTestMessage('floor pad '.repeat(40), 'user', 'floor-pos-u1')];
      const localTokens = sumCountMessage(triggerOm, unobserved);
      expect(localTokens).toBeGreaterThan(0);
      expect(localTokens).toBeLessThanOrEqual(9000);
      for (const message of unobserved) {
        messageList.add(message, 'memory');
      }

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe a second time when the same step usage is replayed after a provider observe', async () => {
      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'stale-obs-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'stale-obs-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      const slack = providerCap - (belowCapUsage.inputTokens + belowCapUsage.outputTokens);
      const resultPart = crossingToolResultPart();
      expect(countSyntheticParts(triggerOm, [resultPart])).toBeGreaterThan(slack);
      messageList.add(
        {
          id: 'asst-stale-observe',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: {
            format: 2,
            parts: [{ type: 'text', text: 'I will look that up after the last model call. '.repeat(20) }, resultPart],
          },
        },
        'response',
      );
      openLoopStepBoundary(messageList);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: belowCapUsage }]));
      expect(triggerOm.observer.call).toHaveBeenCalledTimes(1);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 2, [{ usage: belowCapUsage }]));

      expect(triggerOm.observer.call).toHaveBeenCalledTimes(1);
    });

    it('defers observe while in the async observation band without a ready chunk', async () => {
      const triggerOm = createOccupancyOm({
        messageTokens: 10000,
        bufferTokens: 2000,
        bufferActivation: 0.8,
        blockAfter: 1.5,
        providerContextTokens: providerCap,
      });
      mockObserverCall(triggerOm);
      const record = await triggerOm.getOrCreateRecord(threadId, resourceId);
      vi.spyOn(triggerOm, 'buffer').mockResolvedValue({ buffered: false, record });
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      const bandMessages: MastraDBMessage[] = [];
      let i = 0;
      while (triggerOm.getTokenCounter().countMessages(bandMessages) < 12000) {
        bandMessages.push(
          createTestMessage(`band pad ${i} `.repeat(80), i % 2 ? 'assistant' : 'user', `band-defer-${i}`),
        );
        i += 1;
      }
      const pending = triggerOm.getTokenCounter().countMessages(bandMessages);
      expect(pending).toBeGreaterThanOrEqual(10000);
      expect(pending).toBeLessThan(15000);
      for (const message of bandMessages) {
        messageList.add(message, 'memory');
      }

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('activates a ready chunk without a second sync observe on that step', async () => {
      const triggerOm = createOccupancyOm({
        messageTokens: 10000,
        bufferTokens: 2000,
        bufferActivation: 0.8,
        blockAfter: 1.5,
        providerContextTokens: providerCap,
      });
      mockObserverCall(triggerOm);
      const t0 = Date.now() - 60_000;
      const bandMessages: MastraDBMessage[] = [];
      let i = 0;
      while (triggerOm.getTokenCounter().countMessages(bandMessages) < 12000) {
        const message = createTestMessage(`band pad ${i} `.repeat(80), i % 2 ? 'assistant' : 'user', `band-act-${i}`);
        message.createdAt = new Date(t0 + i * 1000);
        bandMessages.push(message);
        i += 1;
      }
      const pending = triggerOm.getTokenCounter().countMessages(bandMessages);
      expect(pending).toBeGreaterThanOrEqual(10000);
      expect(pending).toBeLessThan(15000);

      const record = await triggerOm.getOrCreateRecord(threadId, resourceId);
      vi.spyOn(triggerOm, 'buffer').mockResolvedValue({ buffered: false, record });
      await storage.saveMessages({ messages: bandMessages.map(message => ({ ...message, threadId, resourceId })) });
      await storage.updateBufferedObservations({
        id: record.id,
        chunk: {
          observations: '- cycle-a observation',
          tokenCount: 20,
          messageIds: bandMessages.map(message => message.id),
          cycleId: 'cycle-a',
          messageTokens: pending,
          lastObservedAt: bandMessages.at(-1)!.createdAt,
        },
      });
      const activateSpy = vi.spyOn(triggerOm, 'activate');
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, rememberedProvider(bandMessages));
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(activateSpy).toHaveBeenCalled();
      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('observes from local added-since in resource scope with sibling-thread storage', async () => {
      const siblingThreadId = 'provider-context-tokens-sibling';
      await storage.saveThread({
        thread: {
          id: siblingThreadId,
          resourceId,
          title: 'Sibling occupancy thread',
          createdAt: new Date(),
          updatedAt: new Date(),
          metadata: {},
        },
      });
      await storage.saveMessages({ messages: createBulkMessages(10, siblingThreadId) });

      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap }, 'resource');
      const otherContext = await triggerOm.getOtherThreadsContext(resourceId, threadId);
      expect(otherContext).toEqual(expect.any(String));
      mockObserverCall(triggerOm);
      const multiSpy = mockObserverMultiThread(triggerOm, threadId);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(createTestMessage('short memory one', 'user', 'res-pos-u1'), 'memory');
      messageList.add(createTestMessage('short memory two', 'assistant', 'res-pos-a1'), 'memory');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(multiSpy).not.toHaveBeenCalled();

      const slack = providerCap - (belowCapUsage.inputTokens + belowCapUsage.outputTokens);
      const resultPart = crossingToolResultPart();
      expect(countSyntheticParts(triggerOm, [resultPart])).toBeGreaterThan(slack);
      messageList.add(
        {
          id: 'asst-resource-pos',
          role: 'assistant',
          type: 'text',
          createdAt: new Date(),
          content: {
            format: 2,
            parts: [{ type: 'text', text: 'I will look that up after the last model call. '.repeat(20) }, resultPart],
          },
        },
        'response',
      );
      openLoopStepBoundary(messageList);

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: belowCapUsage }]));

      expect(multiSpy).toHaveBeenCalled();
      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe in resource scope when only sibling-thread tokens inflate pending occupancy', async () => {
      const siblingThreadId = 'provider-context-tokens-sibling-neg';
      await storage.saveThread({
        thread: {
          id: siblingThreadId,
          resourceId,
          title: 'Sibling occupancy negative',
          createdAt: new Date(),
          updatedAt: new Date(),
          metadata: {},
        },
      });
      await storage.saveMessages({ messages: createBulkMessages(20, siblingThreadId) });

      const triggerOm = createOccupancyOm({ providerContextTokens: providerCap }, 'resource');
      mockObserverCall(triggerOm);
      const multiSpy = mockObserverMultiThread(triggerOm, threadId);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(multiSpy).not.toHaveBeenCalled();
      expect(triggerOm.observer.call).not.toHaveBeenCalled();
    });

    it('does not observe an attachment under-weigh fixture without providerContextTokens', async () => {
      const attachment = createAttachmentUnderweighMessage();
      const probeOm = createOccupancyOm();
      const localSum = probeOm.getTokenCounter().countMessage(attachment);
      expect(localSum).toBeGreaterThan(0);
      const unsetOm = createOccupancyOm({
        messageTokens: localSum + 1000,
        bufferTokens: false,
      });
      mockObserverCall(unsetOm);
      const unsetProcessor = new ObservationalMemoryProcessor(unsetOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(attachment, 'input');

      await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      await unsetProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(unsetOm.observer.call).not.toHaveBeenCalled();
    });

    it('observes the same attachment fixture with providerContextTokens and large per-step usage', async () => {
      const attachment = createAttachmentUnderweighMessage();
      const probeOm = createOccupancyOm({ providerContextTokens: providerCap });
      const localSum = probeOm.getTokenCounter().countMessage(attachment);
      expect(localSum).toBeGreaterThan(0);
      const triggerOm = createOccupancyOm({
        messageTokens: localSum + 1000,
        bufferTokens: false,
        providerContextTokens: providerCap,
      });
      mockObserverCall(triggerOm);
      const triggerProcessor = new ObservationalMemoryProcessor(triggerOm, noopMemoryProvider);
      const state: Record<string, unknown> = {};
      const messageList = new MessageList({ threadId, resourceId });
      messageList.add(attachment, 'input');

      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 0, []));
      expect(triggerOm.observer.call).not.toHaveBeenCalled();
      await triggerProcessor.processInputStep(buildInputStepArgs(messageList, state, 1, [{ usage: overCapUsage }]));

      expect(triggerOm.observer.call).toHaveBeenCalled();
    });
  });
});
