import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

class StubModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];

  async invoke(): Promise<AIMessageChunk> {
    return new AIMessageChunk({ content: 'ok' });
  }
}

const streamConfig = {
  configurable: { thread_id: 'vision-wiring' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

/** Runs one turn and returns the clientOptions the agent model was built with. */
async function clientOptionsFor(
  vision: boolean | undefined
): Promise<Record<string, unknown>> {
  const spy = jest
    .spyOn(init, 'initializeModel')
    .mockReturnValue(new StubModel());
  try {
    const run = await Run.create<t.IState>({
      runId: `vision-${String(vision)}`,
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'only',
            provider: Providers.OPENAI,
            clientOptions: { model: 'text-only', disableStreaming: true },
            ...(vision === undefined ? {} : { vision }),
          },
        ],
        edges: [],
      },
      returnContent: true,
      skipCleanup: true,
    });
    await run.processStream(
      { messages: [new HumanMessage('hi')] },
      streamConfig
    );
    expect(spy).toHaveBeenCalled();
    return spy.mock.calls[0][0].clientOptions as Record<string, unknown>;
  } finally {
    spy.mockRestore();
  }
}

describe('agent vision reaches the model', () => {
  it('passes vision: false so a text-only model gets images stripped', async () => {
    const options = await clientOptionsFor(false);
    expect(options.vision).toBe(false);
    expect(options.model).toBe('text-only');
  });

  it('passes vision: true through', async () => {
    expect((await clientOptionsFor(true)).vision).toBe(true);
  });

  it('leaves clientOptions untouched when the host sets no vision', async () => {
    expect('vision' in (await clientOptionsFor(undefined))).toBe(false);
  });
});
