import {
  AIMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { stripImagesFromMessages } from '@/llm/openai/utils';
import { projectArtifactPayload } from './core';

/**
 * The two halves of image handling on OpenAI-compatible providers have to compose:
 *
 * 1. `projectArtifactPayload` moves a tool's image artifact out of the `tool` message
 *    (which the providers reject) into a following user message, with an assistant bridge
 *    so strict providers accept the role order.
 * 2. `stripImagesFromMessages` then removes images if the target model has no vision.
 *
 * Run in sequence they must leave a payload every provider accepts, and must not lose the
 * tool's own text output on the way - the model still needs to know what the tool returned.
 */
describe('artifact projection composed with the vision strip', () => {
  const imagePart = {
    type: 'image_url' as const,
    image_url: { url: 'data:image/png;base64,AAAA' },
  };

  const toolRun = (): BaseMessage[] => [
    new HumanMessage({ content: 'draw a cat' }),
    new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_1', name: 'image_gen', args: {} }],
    }),
    new ToolMessage({
      content: 'Generated 1 image (512x512, seed 42)',
      tool_call_id: 'call_1',
      artifact: { content: [imagePart] },
    }),
  ];

  const project = (messages: BaseMessage[]) =>
    projectArtifactPayload(messages, 5000, { bridgeUserAfterTool: true });

  const textOf = (message: BaseMessage): string =>
    Array.isArray(message.content)
      ? message.content
        .filter((part) => (part as { type?: string }).type === 'text')
        .map((part) => (part as { text: string }).text)
        .join(' ')
      : String(message.content);

  it('gives a vision model the image, behind an assistant bridge', () => {
    const projected = project(toolRun());
    const sent = stripImagesFromMessages(projected, true);

    expect(sent.map((m) => m.getType())).toEqual([
      'human',
      'ai',
      'tool',
      'ai',
      'human',
    ]);
    const last = sent[sent.length - 1];
    expect(last.content).toContainEqual(imagePart);
    expect(textOf(last)).toContain('Generated 1 image');
  });

  it('keeps the tool text for a non-vision model and drops only the image', () => {
    const projected = project(toolRun());
    const sent = stripImagesFromMessages(projected, false);

    /** Role order is unchanged, so a strict provider still accepts the request. */
    expect(sent.map((m) => m.getType())).toEqual([
      'human',
      'ai',
      'tool',
      'ai',
      'human',
    ]);

    const last = sent[sent.length - 1];
    const parts = last.content as Array<{ type?: string }>;
    expect(parts.some((p) => p.type === 'image_url')).toBe(false);
    /** The tool's own output must survive - it is the model's only record of the result. */
    expect(textOf(last)).toContain('Generated 1 image');
  });

  it('never places a user message directly after a tool message', () => {
    for (const visionCapable of [true, false]) {
      const sent = stripImagesFromMessages(project(toolRun()), visionCapable);
      const roles = sent.map((m) => m.getType());
      for (let i = 1; i < roles.length; i++) {
        expect(`${roles[i - 1]}->${roles[i]}`).not.toBe('tool->human');
      }
    }
  });
});
