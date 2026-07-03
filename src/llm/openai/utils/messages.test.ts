import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  _convertMessagesToOpenAIParams,
  stripImagesFromMessages,
} from './index';

describe('_convertMessagesToOpenAIParams', () => {
  it('includes reasoning_content for assistant messages in tool-call context when requested', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
      new ToolMessage({
        content: '57531',
        tool_call_id: 'call_1',
      }),
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Calculator returned 57531.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(3);
    expect(params[0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'Need calculator.',
      })
    );
    expect(params[2]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        reasoning_content: 'Calculator returned 57531.',
      })
    );
  });

  it('does not include reasoning_content for no-tool assistant messages', () => {
    const messages = [
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Mental calculation.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(1);
    expect(params[0]).not.toHaveProperty('reasoning_content');
  });

  it('does not include reasoning_content unless explicitly requested', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro');

    expect(params).toHaveLength(1);
    expect(params[0]).not.toHaveProperty('reasoning_content');
  });

  it('keeps reasoning_content latched after tool-call context is established', () => {
    const messages = [
      new AIMessage({
        content: 'No tool was needed.',
        additional_kwargs: {
          reasoning_content: 'Initial no-tool reasoning.',
        },
      }),
      new HumanMessage('Use the calculator.'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
      new ToolMessage({
        content: '57531',
        tool_call_id: 'call_1',
      }),
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Calculator returned 57531.',
        },
      }),
      new HumanMessage('Was that correct?'),
      new AIMessage({
        content: 'Yes.',
        additional_kwargs: {
          reasoning_content: 'The prior calculator result is available.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(7);
    expect(params[0]).not.toHaveProperty('reasoning_content');
    expect(params[2]).toEqual(
      expect.objectContaining({
        reasoning_content: 'Need calculator.',
      })
    );
    expect(params[4]).toEqual(
      expect.objectContaining({
        reasoning_content: 'Calculator returned 57531.',
      })
    );
    expect(params[6]).toEqual(
      expect.objectContaining({
        reasoning_content: 'The prior calculator result is available.',
      })
    );
  });
});

describe('stripImagesFromMessages', () => {
  const textPart = { type: 'text' as const, text: 'Is this image real?' };
  const imageUrlPart = {
    type: 'image_url' as const,
    image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
  };
  // A LangChain standard data content block (how uploaded images can arrive).
  const imageDataBlock = {
    type: 'image' as const,
    source_type: 'base64' as const,
    mime_type: 'image/png',
    data: 'iVBORw0KGgo=',
  };

  it('removes image_url parts for non-vision models', () => {
    const [msg] = stripImagesFromMessages(
      [new HumanMessage({ content: [textPart, imageUrlPart] })],
      false
    );
    expect(msg.content).toEqual([textPart]);
  });

  it('removes image data content blocks for non-vision models', () => {
    const [msg] = stripImagesFromMessages(
      [new HumanMessage({ content: [textPart, imageDataBlock] })],
      false
    );
    expect(msg.content).toEqual([textPart]);
  });

  it('substitutes a placeholder when only image content remains', () => {
    const [msg] = stripImagesFromMessages(
      [new HumanMessage({ content: [imageDataBlock] })],
      false
    );
    expect(msg.content).toEqual([
      { type: 'text', text: expect.stringContaining('Image content omitted') },
    ]);
  });

  it('leaves messages untouched for vision-capable models', () => {
    const content = [textPart, imageDataBlock];
    const [msg] = stripImagesFromMessages(
      [new HumanMessage({ content })],
      true
    );
    expect(msg.content).toEqual(content);
  });
});
