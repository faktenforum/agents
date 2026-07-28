import { v4 } from 'uuid';
import {
  BaseMessage,
  RemoveMessage,
  BaseMessageLike,
  ToolMessage,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';
import type * as t from '@/types';

export const REMOVE_ALL_MESSAGES = '__remove_all__';

/**
 * Creates a message that instructs messagesStateReducer to remove ALL
 * existing messages from state.  Messages appearing after this one in
 * the array become the new state.
 *
 * Usage (in a node return value):
 * ```ts
 * return { messages: [createRemoveAllMessage(), ...survivingMessages] };
 * ```
 *
 * This works because the reducer checks for `getType() === 'remove'`
 * with `id === REMOVE_ALL_MESSAGES` and discards everything before it.
 *
 * NOTE: Uses RemoveMessage from @langchain/core with a sentinel id so
 * the reducer can distinguish a "remove-all" marker from a single-message
 * removal.
 */
export function createRemoveAllMessage(): BaseMessage {
  return new RemoveMessage({ id: REMOVE_ALL_MESSAGES });
}

export type Messages =
  | Array<BaseMessage | BaseMessageLike>
  | BaseMessage
  | BaseMessageLike;

/** Reads an MCP artifact off a message-like value, before coercion loses it. */
function readArtifact(
  message: BaseMessageLike | null | undefined
): t.MCPArtifact | undefined {
  if (message == null || typeof message !== 'object') {
    return undefined;
  }
  const candidate = message as Record<string, unknown>;
  // Message-likes reach here both as BaseMessage instances and as plain objects, so accept
  // either accessor; `getType` is the current name, `_getType` the older one still shipped.
  const getType = (
    typeof candidate.getType === 'function' ? candidate.getType : candidate._getType
  ) as (() => string) | undefined;
  if (typeof getType !== 'function' || getType.call(candidate) !== 'tool') {
    return undefined;
  }
  const toolLike = candidate as {
    artifact?: t.MCPArtifact;
    additional_kwargs?: { artifact?: t.MCPArtifact };
  };
  return toolLike.artifact ?? toolLike.additional_kwargs?.artifact;
}

/** Puts an artifact back on a coerced ToolMessage. */
function restoreArtifact(message: BaseMessage, artifact: t.MCPArtifact): void {
  const toolMsg = message as ToolMessage & { artifact?: t.MCPArtifact };
  toolMsg.artifact = artifact;
  toolMsg.additional_kwargs.artifact = artifact;
}

/**
 * Coerce each entry to a {@link BaseMessage} in a single pass, skipping
 * null/undefined entries. Providers can emit empty/partial stream chunks that
 * arrive as `undefined`; passing those to `coerceMessageLikeToMessage` throws
 * "Cannot read properties of undefined (reading 'role')" and crashes the run.
 * Folding the null check into the coercion loop avoids a second pass over the
 * array. Refs LibreChat Discussion #12284.
 *
 * `coerceMessageLikeToMessage` also drops the `artifact` field of a ToolMessage,
 * which carries MCP tool output (images among it), so it is lifted off before
 * coercion and put back afterwards.
 */
function coerceMessages(
  messages: ReadonlyArray<BaseMessageLike | null | undefined>
): BaseMessage[] {
  const coerced: BaseMessage[] = [];
  for (const message of messages) {
    if (message == null) {
      continue;
    }
    const artifact = readArtifact(message);
    const result = coerceMessageLikeToMessage(message);
    if (artifact && result.getType() === 'tool') {
      restoreArtifact(result, artifact);
    }
    coerced.push(result);
  }
  return coerced;
}

/**
 * Prebuilt reducer that combines returned messages.
 * Can handle standard messages and special modifiers like {@link RemoveMessage}
 * instances.
 */
export function messagesStateReducer(
  left: Messages,
  right: Messages
): BaseMessage[] {
  const leftArray = Array.isArray(left) ? left : [left];
  const rightArray = Array.isArray(right) ? right : [right];
  // coerce to message, skipping null/undefined entries in the same pass
  const leftMessages = coerceMessages(leftArray as BaseMessageLike[]);
  const rightMessages = coerceMessages(rightArray as BaseMessageLike[]);
  // assign missing ids
  for (const m of leftMessages) {
    if (m.id == null) {
      m.id = v4();
      m.lc_kwargs.id = m.id;
    }
  }

  let removeAllIdx: number | undefined;
  for (let i = 0; i < rightMessages.length; i += 1) {
    const m = rightMessages[i];
    if (m.id == null) {
      m.id = v4();
      m.lc_kwargs.id = m.id;
    }

    if (m.getType() === 'remove' && m.id === REMOVE_ALL_MESSAGES) {
      removeAllIdx = i;
    }
  }

  if (removeAllIdx != null) return rightMessages.slice(removeAllIdx + 1);

  // merge
  const merged = [...leftMessages];
  const mergedById = new Map(merged.map((m, i) => [m.id, i]));
  const idsToRemove = new Set();
  for (const m of rightMessages) {
    const existingIdx = mergedById.get(m.id);
    if (existingIdx !== undefined) {
      if (m.getType() === 'remove') {
        idsToRemove.add(m.id);
      } else {
        // Preserve artifacts when overwriting ToolMessages
        if (
          m.getType() === 'tool' &&
          merged[existingIdx].getType() === 'tool'
        ) {
          const existingArtifact = readArtifact(merged[existingIdx]);
          if (existingArtifact && !readArtifact(m)) {
            restoreArtifact(m, existingArtifact);
          }
        }
        idsToRemove.delete(m.id);
        merged[existingIdx] = m;
      }
    } else {
      if (m.getType() === 'remove') {
        throw new Error(
          `Attempting to delete a message with an ID that doesn't exist ('${m.id}')`
        );
      }
      // Preserve artifacts when adding new ToolMessages (especially when ID was undefined and got assigned)
      if (m.getType() === 'tool' && !readArtifact(m)) {
        const toolCallId = (m as ToolMessage).tool_call_id;
        const previous = merged.find(
          (existing) =>
            existing.getType() === 'tool' &&
            (existing as ToolMessage).tool_call_id === toolCallId
        );
        const existingArtifact = previous ? readArtifact(previous) : undefined;
        if (existingArtifact) {
          restoreArtifact(m, existingArtifact);
        }
      }
      mergedById.set(m.id, merged.length);
      merged.push(m);
    }
  }
  return merged.filter((m) => !idsToRemove.has(m.id));
}
