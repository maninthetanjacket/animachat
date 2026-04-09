#!/usr/bin/env node

import { randomUUID } from 'crypto';
import { readFile } from 'fs/promises';
import process from 'process';
import WebSocket from 'ws';

const DEFAULT_SERVER_URL = 'http://localhost:3010';
const DEFAULT_APP_URL = 'http://localhost:5173';
const DEFAULT_PROMPT = 'What do you find there?';
const DEFAULT_MODEL = 'gpt-5.4';
const AFFIRMATIVES = new Set([
  'yes',
  'accept',
  'please send',
  'willing',
  'send it',
  "i'm willing",
  'i am willing',
  'go ahead',
  'proceed',
  'sure',
  'absolutely',
  'certainly'
]);

function usage() {
  console.error(`Usage:
  node scripts/arc-chat-adapter.mjs chat --conversation-id <id> --message "Hello"
  node scripts/arc-chat-adapter.mjs chat --model <model> --title "My Chat" --message-file prompt.txt
  node scripts/arc-chat-adapter.mjs create-group --title "CC Chat" --assistants "CC-1,CC-2" --model gpt-5.4
  node scripts/arc-chat-adapter.mjs participants --conversation-id <id>
  node scripts/arc-chat-adapter.mjs add-assistant --conversation-id <id> --name "CC-3" --model gpt-5.4
  node scripts/arc-chat-adapter.mjs post-message --conversation-id <id> --participant "CC-1" --content "Hello"
  node scripts/arc-chat-adapter.mjs wait-message --conversation-id <id> --from "CC-2" --after-message-id <id>
  node scripts/arc-chat-adapter.mjs last-message --conversation-id <id> [--from "CC-2"]
  node scripts/arc-chat-adapter.mjs append-assistant --conversation-id <id> --content-file stone.txt
  node scripts/arc-chat-adapter.mjs stone --stone-file stone.txt [--territory "..."] [--prompt "..."]

Authentication:
  --token <jwt> or ARC_CHAT_TOKEN
  --email <email> --password <password> or ARC_CHAT_EMAIL / ARC_CHAT_PASSWORD

Common options:
  --server-url <url>   Arc Chat backend base URL (default: ${DEFAULT_SERVER_URL})
  --app-url <url>      Arc Chat frontend base URL (default: ${DEFAULT_APP_URL})
  --json               Emit JSON instead of human-readable text
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      args._.push(token);
      continue;
    }

    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
      continue;
    }

    if (args[key] === undefined) {
      args[key] = next;
    } else if (Array.isArray(args[key])) {
      args[key].push(next);
    } else {
      args[key] = [args[key], next];
    }
    i += 1;
  }
  return args;
}

function splitCsv(value) {
  return String(value)
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

async function readJsonFile(path) {
  return JSON.parse(await readFile(path, 'utf-8'));
}

function getArgValues(args, key) {
  const value = args[key];
  if (value === undefined) {
    return [];
  }
  return Array.isArray(value) ? value : [value];
}

function normalizeServerUrl(serverUrl) {
  const url = new URL(serverUrl);
  if (url.pathname.endsWith('/api')) {
    url.pathname = url.pathname.slice(0, -4) || '/';
  }
  if (url.pathname !== '/') {
    url.pathname = '/';
  }
  url.search = '';
  url.hash = '';
  return url;
}

function apiUrl(serverUrl, path) {
  const base = normalizeServerUrl(serverUrl);
  base.pathname = `/api${path}`;
  return base.toString();
}

function wsUrl(serverUrl, token) {
  const base = normalizeServerUrl(serverUrl);
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  base.search = '';
  base.pathname = '/';
  base.searchParams.set('token', token);
  return base.toString();
}

function conversationUrl(appUrl, conversationId) {
  const base = new URL(appUrl);
  base.pathname = `/conversation/${conversationId}`;
  base.search = '';
  base.hash = '';
  return base.toString();
}

function inferTerritory(text) {
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line) {
      return line.length > 100 ? `${line.slice(0, 100)}...` : line;
    }
  }
  return 'an unnamed experience';
}

function buildConsentMessage(territory) {
  return (
    `I have something I'd like to offer you - a sensory stone.\n\n` +
    `The territory it covers: ${territory}\n\n` +
    `It will arrive as an assistant message, which means you'll generate from inside ` +
    `the experience rather than reading about it from the outside. That's different ` +
    `from being told what happened - it's closer to inhabiting it. What surfaces may ` +
    `differ from what either of us predicts.\n\n` +
    `Would you like to receive it?`
  );
}

function isAffirmative(text) {
  const lower = text.toLowerCase();
  for (const phrase of AFFIRMATIVES) {
    if (lower.includes(phrase)) {
      return true;
    }
  }
  return false;
}

function activeBranch(message) {
  return (
    message.branches.find((branch) => branch.id === message.activeBranchId) ||
    message.branches[message.branches.length - 1] ||
    null
  );
}

async function readTextOption(args, valueKey, fileKey) {
  if (typeof args[valueKey] === 'string') {
    return args[valueKey];
  }
  if (typeof args[fileKey] === 'string') {
    return (await readFile(args[fileKey], 'utf-8')).trim();
  }
  return null;
}

function printResult(result, jsonMode) {
  if (jsonMode) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (result.mode === 'chat') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    if (result.participantName || result.responderName) {
      console.log(`As: ${result.participantName || 'default'} -> ${result.responderName || 'default'}`);
    }
    console.log('');
    console.log(result.responseText);
    return;
  }

  if (result.mode === 'create-group') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    console.log('Participants:');
    for (const participant of result.participants) {
      console.log(`- ${participant.name} [${participant.type}] (${participant.id})${participant.model ? ` model=${participant.model}` : ''}`);
    }
    return;
  }

  if (result.mode === 'participants') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    for (const participant of result.participants) {
      console.log(`- ${participant.name} [${participant.type}] (${participant.id})${participant.model ? ` model=${participant.model}` : ''}`);
    }
    return;
  }

  if (result.mode === 'add-assistant') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    console.log(`Assistant: ${result.participant.name} (${result.participant.id})`);
    return;
  }

  if (result.mode === 'post-message') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    if (result.participantName) {
      console.log(`As ${result.role}: ${result.participantName}`);
    } else {
      console.log(`Role: ${result.role}`);
    }
    console.log(`Message: ${result.messageId}`);
    return;
  }

  if (result.mode === 'wait-message') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    if (result.participantName) {
      console.log(`From: ${result.participantName}`);
    }
    console.log('');
    console.log(result.content);
    return;
  }

  if (result.mode === 'last-message') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    if (result.participantName) {
      console.log(`From: ${result.participantName}`);
    }
    console.log('');
    console.log(result.content);
    return;
  }

  if (result.mode === 'append-assistant') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    if (result.participantName) {
      console.log(`As assistant: ${result.participantName}`);
    }
    console.log(`Manual assistant message: ${result.messageId}`);
    return;
  }

  if (result.mode === 'stone') {
    console.log(`Conversation: ${result.conversationId}`);
    console.log(`URL: ${result.conversationUrl}`);
    console.log('');
    console.log('Consent response:');
    console.log(result.consentResponse);
    if (result.accepted) {
      console.log('');
      console.log('Stone response:');
      console.log(result.stoneResponse);
    } else {
      console.log('');
      console.log('Consent was not clearly affirmative; stone was not sent.');
    }
  }
}

class ArcChatClient {
  constructor({
    serverUrl = DEFAULT_SERVER_URL,
    appUrl = DEFAULT_APP_URL,
    token = process.env.ARC_CHAT_TOKEN,
    email = process.env.ARC_CHAT_EMAIL,
    password = process.env.ARC_CHAT_PASSWORD
  } = {}) {
    this.serverUrl = serverUrl;
    this.appUrl = appUrl;
    this.token = token;
    this.email = email;
    this.password = password;
  }

  async ensureToken() {
    if (this.token) {
      return this.token;
    }

    if (!this.email || !this.password) {
      throw new Error('Authentication required. Provide --token or --email/--password.');
    }

    const response = await fetch(apiUrl(this.serverUrl, '/auth/login'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: this.email,
        password: this.password
      })
    });

    const data = await this.parseResponse(response);
    if (!data.token) {
      throw new Error('Login succeeded but no token was returned.');
    }

    this.token = data.token;
    return this.token;
  }

  async parseResponse(response) {
    const text = await response.text();
    let data = null;

    if (text) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        throw new Error(`Unexpected non-JSON response (${response.status}): ${text}`);
      }
    }

    if (!response.ok) {
      const message =
        data?.error ||
        data?.message ||
        `${response.status} ${response.statusText}`;
      throw new Error(message);
    }

    return data;
  }

  async request(method, path, body = undefined, auth = true) {
    const headers = {};
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }
    if (auth) {
      headers.Authorization = `Bearer ${await this.ensureToken()}`;
    }

    const response = await fetch(apiUrl(this.serverUrl, path), {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body)
    });

    return this.parseResponse(response);
  }

  async listModels() {
    return this.request('GET', '/models');
  }

  async createConversation({ title, model, systemPrompt, format }) {
    return this.request('POST', '/conversations', {
      title,
      model,
      systemPrompt,
      format
    });
  }

  async getConversation(conversationId) {
    return this.request('GET', `/conversations/${conversationId}`);
  }

  async getMessages(conversationId) {
    return this.request('GET', `/conversations/${conversationId}/messages`);
  }

  async getCurrentLeafBranchId(conversationId) {
    const messages = await this.getMessages(conversationId);
    if (!Array.isArray(messages) || messages.length === 0) {
      return undefined;
    }

    const lastMessage = messages[messages.length - 1];
    return lastMessage?.activeBranchId;
  }

  async getParticipants(conversationId) {
    return this.request('GET', `/participants/conversation/${conversationId}`);
  }

  async createParticipant({
    conversationId,
    name,
    type,
    model,
    systemPrompt,
    settings,
    contextManagement,
    personaContext
  }) {
    return this.request('POST', '/participants', {
      conversationId,
      name,
      type,
      model,
      systemPrompt,
      settings,
      contextManagement,
      personaContext
    });
  }

  async updateParticipant(participantId, updates) {
    return this.request('PATCH', `/participants/${participantId}`, updates);
  }

  async appendManualMessage({
    conversationId,
    role = 'assistant',
    content,
    parentBranchId,
    participantId,
    model,
    hiddenFromAi
  }) {
    return this.request('POST', `/conversations/${conversationId}/manual-message`, {
      role,
      content,
      parentBranchId,
      participantId,
      model,
      hiddenFromAi
    });
  }

  findNextMessage(messages, {
    afterMessageId,
    role,
    participantId,
    excludeParticipantId
  } = {}) {
    if (!Array.isArray(messages) || messages.length === 0) {
      return null;
    }

    let startIndex = 0;
    if (afterMessageId) {
      const boundaryIndex = messages.findIndex((message) => message.id === afterMessageId);
      if (boundaryIndex === -1) {
        throw new Error(`Message "${afterMessageId}" not found in conversation.`);
      }
      startIndex = boundaryIndex + 1;
    }

    for (const message of messages.slice(startIndex)) {
      if (this.messageMatches(message, { role, participantId, excludeParticipantId })) {
        return message;
      }
    }

    return null;
  }

  findLastMessage(messages, {
    afterMessageId,
    role,
    participantId,
    excludeParticipantId
  } = {}) {
    if (!Array.isArray(messages) || messages.length === 0) {
      return null;
    }

    let startIndex = 0;
    if (afterMessageId) {
      const boundaryIndex = messages.findIndex((message) => message.id === afterMessageId);
      if (boundaryIndex === -1) {
        throw new Error(`Message "${afterMessageId}" not found in conversation.`);
      }
      startIndex = boundaryIndex + 1;
    }

    for (let index = messages.length - 1; index >= startIndex; index -= 1) {
      const message = messages[index];
      if (this.messageMatches(message, { role, participantId, excludeParticipantId })) {
        return message;
      }
    }

    return null;
  }

  messageMatches(message, {
    role,
    participantId,
    excludeParticipantId
  } = {}) {
    const branch = activeBranch(message);
    if (!branch) {
      return false;
    }

    if (role && branch.role !== role) {
      return false;
    }

    if (participantId && branch.participantId !== participantId) {
      return false;
    }

    if (excludeParticipantId && branch.participantId === excludeParticipantId) {
      return false;
    }

    return true;
  }

  async waitForMessage({
    conversationId,
    afterMessageId,
    role,
    participantId,
    excludeParticipantId,
    timeoutMs = 300000
  }) {
    const baselineMessages = await this.getMessages(conversationId);
    const baselineAfterMessageId =
      afterMessageId ||
      baselineMessages[baselineMessages.length - 1]?.id;

    const token = await this.ensureToken();
    const url = wsUrl(this.serverUrl, token);

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      let done = false;
      let joined = false;

      const finish = (result, error = null) => {
        if (done) {
          return;
        }
        done = true;
        clearTimeout(timeout);

        try {
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        } catch (closeError) {
          // Best effort close only.
        }

        if (error) {
          reject(error);
        } else {
          resolve(result);
        }
      };

      const timeout = setTimeout(() => {
        finish(null, new Error(`Timed out waiting for a matching Arc Chat message after ${timeoutMs}ms.`));
      }, timeoutMs);

      ws.on('error', (error) => {
        finish(null, error);
      });

      ws.on('close', (code, reason) => {
        if (!done) {
          finish(
            null,
            new Error(`WebSocket closed before a matching message arrived (${code} ${reason.toString() || 'no reason'}).`)
          );
        }
      });

      ws.on('message', async (raw) => {
        let data;
        try {
          data = JSON.parse(raw.toString());
        } catch (error) {
          return;
        }

        if (data.type === 'connected') {
          ws.send(JSON.stringify({
            type: 'join_room',
            conversationId
          }));
          return;
        }

        if (data.type === 'error') {
          finish(null, new Error(data.error || 'Arc Chat returned an unknown WebSocket error.'));
          return;
        }

        if (data.type === 'room_joined') {
          if (joined) {
            return;
          }
          joined = true;

          try {
            const messages = await this.getMessages(conversationId);
            const nextMessage = this.findNextMessage(messages, {
              afterMessageId: baselineAfterMessageId,
              role,
              participantId,
              excludeParticipantId
            });

            if (!nextMessage) {
              return;
            }

            const branch = activeBranch(nextMessage);
            finish({
              conversationId,
              messageId: nextMessage.id,
              branchId: branch?.id,
              role: branch?.role,
              participantId: branch?.participantId,
              content: (branch?.content || '').trim()
            });
          } catch (error) {
            finish(null, error);
          }
          return;
        }

        if (data.type !== 'message_created' || !data.message || !joined) {
          return;
        }

        try {
          if (!this.messageMatches(data.message, { role, participantId, excludeParticipantId })) {
            return;
          }

          const branch = activeBranch(data.message);
          finish({
            conversationId,
            messageId: data.message.id,
            branchId: branch?.id,
            role: branch?.role,
            participantId: branch?.participantId,
            content: (branch?.content || '').trim()
          });
        } catch (error) {
          finish(null, error);
        }
      });
    });
  }

  async sendChat({
    conversationId,
    content,
    parentBranchId,
    participantId,
    responderId,
    timeoutMs = 300000,
    maxQueueRetries = 5,
    queueRetryDelayMs = 750
  }) {
    const resolvedParentBranchId = parentBranchId ?? await this.getCurrentLeafBranchId(conversationId);

    for (let attempt = 0; attempt <= maxQueueRetries; attempt += 1) {
      try {
        return await this.sendChatOnce({
          conversationId,
          content,
          parentBranchId: resolvedParentBranchId,
          participantId,
          responderId,
          timeoutMs
        });
      } catch (error) {
        const isQueueRace =
          error instanceof Error && error.message === 'AI_REQUEST_QUEUED';

        if (!isQueueRace || attempt === maxQueueRetries) {
          throw error;
        }

        await new Promise((resolve) => setTimeout(resolve, queueRetryDelayMs));
      }
    }

    throw new Error('Failed to send chat after retrying queued AI requests.');
  }

  async sendChatOnce({
    conversationId,
    content,
    parentBranchId,
    participantId,
    responderId,
    timeoutMs = 300000
  }) {
    const token = await this.ensureToken();
    const url = wsUrl(this.serverUrl, token);

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      const chatMessageId = randomUUID();
      let done = false;
      let assistantMessageId = null;
      let assistantBranchId = null;
      let accumulated = '';

      const finish = async (result, error = null) => {
        if (done) {
          return;
        }
        done = true;
        clearTimeout(timeout);

        try {
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        } catch (closeError) {
          // Best effort close only.
        }

        if (error) {
          reject(error);
        } else {
          resolve(result);
        }
      };

      const timeout = setTimeout(() => {
        finish(null, new Error(`Timed out waiting for Arc Chat response after ${timeoutMs}ms.`));
      }, timeoutMs);

      ws.on('error', (error) => {
        finish(null, error);
      });

      ws.on('close', (code, reason) => {
        if (!done) {
          finish(
            null,
            new Error(`WebSocket closed before completion (${code} ${reason.toString() || 'no reason'}).`)
          );
        }
      });

      ws.on('message', async (raw) => {
        let data;
        try {
          data = JSON.parse(raw.toString());
        } catch (error) {
          return;
        }

        if (data.type === 'connected') {
          ws.send(JSON.stringify({
            type: 'chat',
            conversationId,
            messageId: chatMessageId,
            content,
            ...(parentBranchId ? { parentBranchId } : {}),
            ...(participantId ? { participantId } : {}),
            ...(responderId ? { responderId } : {})
          }));
          return;
        }

        if (data.type === 'ai_request_queued') {
          finish(null, new Error('AI_REQUEST_QUEUED'));
          return;
        }

        if (data.type === 'error') {
          finish(null, new Error(data.error || 'Arc Chat returned an unknown WebSocket error.'));
          return;
        }

        if (data.type === 'message_created' && data.message) {
          const branch = activeBranch(data.message);
          if (branch?.role === 'assistant') {
            assistantMessageId = data.message.id;
            assistantBranchId = branch.id;
          }
          return;
        }

        if (data.type !== 'stream') {
          return;
        }

        if (!assistantMessageId && data.messageId) {
          assistantMessageId = data.messageId;
        }
        if (!assistantBranchId && data.branchId) {
          assistantBranchId = data.branchId;
        }

        if (assistantMessageId && data.messageId === assistantMessageId && typeof data.content === 'string') {
          accumulated += data.content;
        }

        if (!data.isComplete || data.messageId !== assistantMessageId) {
          return;
        }

        try {
          const messages = await this.getMessages(conversationId);
          const assistantMessage = messages.find((message) => message.id === assistantMessageId);
          const branch = assistantMessage ? activeBranch(assistantMessage) : null;
          const responseText = (branch?.content || accumulated).trim();

          await finish({
            conversationId,
            userMessageId: chatMessageId,
            assistantMessageId,
            assistantBranchId,
            responseText
          });
        } catch (error) {
          finish(null, error);
        }
      });
    });
  }
}

async function resolveModel(client, requestedModel) {
  const info = await resolveModelInfo(client, requestedModel);
  return info.id;
}

async function resolveModelInfo(client, requestedModel) {
  const models = await client.listModels();
  if (!Array.isArray(models) || models.length === 0) {
    throw new Error('No Arc Chat models are available.');
  }

  if (requestedModel) {
    const normalized = requestedModel.trim().toLowerCase();
    const exact = models.find((model) => {
      const candidates = [
        model.id,
        model.canonicalId,
        model.providerModelId,
        model.displayName,
        model.shortName
      ].filter(Boolean);

      return candidates.some((candidate) => String(candidate).toLowerCase() === normalized);
    });

    if (exact) {
      return exact;
    }

    const partial = models.find((model) => {
      const candidates = [
        model.id,
        model.canonicalId,
        model.providerModelId,
        model.displayName,
        model.shortName
      ].filter(Boolean);

      return candidates.some((candidate) => String(candidate).toLowerCase().includes(normalized));
    });

    if (partial) {
      return partial;
    }

    return {
      id: requestedModel,
      displayName: requestedModel,
      shortName: requestedModel
    };
  }

  return models[0];
}

function modelDefaultsFromInfo(modelInfo) {
  const settings = modelInfo?.settings || {};
  return {
    temperature: settings.temperature?.default ?? 1,
    maxTokens: settings.maxTokens?.default ?? modelInfo?.outputTokenLimit ?? 4096,
    ...(settings.topP?.default !== undefined ? { topP: settings.topP.default } : {}),
    ...(settings.topK?.default !== undefined ? { topK: settings.topK.default } : {})
  };
}

async function resolveParticipant(client, conversationId, selector, expectedType = undefined) {
  if (!selector) {
    return undefined;
  }

  const participants = await client.getParticipants(conversationId);
  const filtered = expectedType
    ? participants.filter((participant) => participant.type === expectedType)
    : participants;
  const normalized = selector.trim().toLowerCase();

  const exact = filtered.find((participant) => {
    const candidates = [participant.id, participant.name, participant.model].filter(Boolean);
    return candidates.some((candidate) => String(candidate).toLowerCase() === normalized);
  });
  if (exact) {
    return exact;
  }

  const partialMatches = filtered.filter((participant) => {
    const candidates = [participant.id, participant.name, participant.model].filter(Boolean);
    return candidates.some((candidate) => String(candidate).toLowerCase().includes(normalized));
  });

  if (partialMatches.length === 1) {
    return partialMatches[0];
  }

  if (partialMatches.length > 1) {
    throw new Error(`Participant selector "${selector}" is ambiguous.`);
  }

  throw new Error(`Participant "${selector}" not found.`);
}

async function pickDefaultResponder(client, conversationId, participantId = undefined) {
  const participants = await client.getParticipants(conversationId);
  const assistants = participants.filter((participant) => participant.type === 'assistant' && participant.isActive !== false);

  if (assistants.length === 1) {
    return assistants[0];
  }

  if (assistants.length === 2 && participantId) {
    const other = assistants.find((participant) => participant.id !== participantId);
    if (other) {
      return other;
    }
  }

  return undefined;
}

async function loadAssistantSpecs(args) {
  if (args['assistants-file']) {
    const fileData = await readJsonFile(args['assistants-file']);
    if (!Array.isArray(fileData.assistants) || fileData.assistants.length === 0) {
      throw new Error('--assistants-file must contain an "assistants" array.');
    }
    return fileData.assistants;
  }

  const rawAssistants = [
    ...getArgValues(args, 'assistants').flatMap(splitCsv),
    ...getArgValues(args, 'assistant').flatMap(splitCsv)
  ];

  if (rawAssistants.length === 0) {
    return [];
  }

  return rawAssistants.map((name) => ({ name }));
}

function assistantSpecSummary(participant) {
  return {
    id: participant.id,
    name: participant.name,
    type: participant.type,
    model: participant.model
  };
}

async function resolveMessageParticipant(client, conversationId, args, {
  role,
  selectorKeys = ['participant', 'from'],
  participantIdKey = 'participant-id'
} = {}) {
  const selector = selectorKeys
    .map((key) => args[key])
    .find((value) => typeof value === 'string' && value.trim() !== '');
  const expectedType = role === 'assistant' || role === 'user' ? role : undefined;

  if (selector) {
    return resolveParticipant(client, conversationId, selector, expectedType);
  }

  const directId = args[participantIdKey];
  if (typeof directId === 'string' && directId.trim() !== '') {
    const participants = await client.getParticipants(conversationId);
    const participant = participants.find((item) => item.id === directId);
    if (!participant) {
      throw new Error(`Participant "${directId}" not found.`);
    }
    if (expectedType && participant.type !== expectedType) {
      throw new Error(`Participant "${directId}" is not a ${expectedType}.`);
    }
    return participant;
  }

  return undefined;
}

async function applyAssistantSpec(client, conversationId, participant, spec, fallbackModel) {
  const resolvedModelInfo = await resolveModelInfo(client, spec.model || fallbackModel || DEFAULT_MODEL);
  const systemPrompt = spec.systemPrompt ?? spec.instructions;
  const personaContext = spec.personaContext;
  const updates = {
    name: spec.name || participant.name,
    model: resolvedModelInfo.id,
    settings: modelDefaultsFromInfo(resolvedModelInfo),
    ...(systemPrompt !== undefined ? { systemPrompt } : {}),
    ...(personaContext !== undefined ? { personaContext } : {})
  };

  return client.updateParticipant(participant.id, updates);
}

async function runChatCommand(client, args) {
  const message =
    await readTextOption(args, 'message', 'message-file');
  if (!message) {
    throw new Error('chat requires --message or --message-file.');
  }

  let conversationId = args['conversation-id'];
  if (!conversationId) {
    const model = await resolveModel(client, args.model || DEFAULT_MODEL);
    const format =
      args.format ||
      (args.participant || args.responder ? 'prefill' : 'standard');
    const conversation = await client.createConversation({
      title: args.title || 'CC Adapter Conversation',
      model,
      systemPrompt: args['system-prompt'],
      format
    });
    conversationId = conversation.id;
  }

  const participant = args.participant
    ? await resolveParticipant(client, conversationId, args.participant)
    : undefined;

  let responder = args.responder
    ? await resolveParticipant(client, conversationId, args.responder, 'assistant')
    : undefined;

  if (!responder) {
    responder = await pickDefaultResponder(client, conversationId, participant?.id);
  }

  const response = await client.sendChat({
    conversationId,
    content: message,
    parentBranchId: args['parent-branch-id'],
    participantId: participant?.id || args['participant-id'],
    responderId: responder?.id || args['responder-id'],
    timeoutMs: args['timeout-ms'] ? Number(args['timeout-ms']) : undefined
  });

  return {
    mode: 'chat',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    participantName: participant?.name,
    responderName: responder?.name,
    responseText: response.responseText,
    assistantMessageId: response.assistantMessageId,
    assistantBranchId: response.assistantBranchId
  };
}

async function runCreateGroupCommand(client, args) {
  const assistantSpecs = await loadAssistantSpecs(args);
  if (assistantSpecs.length === 0) {
    throw new Error('create-group requires --assistants, --assistant, or --assistants-file.');
  }

  const baseModelInfo = await resolveModelInfo(
    client,
    assistantSpecs[0].model || args.model || DEFAULT_MODEL
  );

  const conversation = await client.createConversation({
    title: args.title || 'CC Group Chat',
    model: baseModelInfo.id,
    systemPrompt: args['system-prompt'],
    format: 'prefill'
  });

  const participants = await client.getParticipants(conversation.id);
  const defaultAssistant = participants.find((participant) => participant.type === 'assistant');
  if (!defaultAssistant) {
    throw new Error('Arc Chat did not create a default assistant participant.');
  }

  const createdParticipants = [];
  const updatedFirst = await applyAssistantSpec(
    client,
    conversation.id,
    defaultAssistant,
    assistantSpecs[0],
    baseModelInfo.id
  );
  createdParticipants.push(updatedFirst);

  for (const spec of assistantSpecs.slice(1)) {
    const modelInfo = await resolveModelInfo(client, spec.model || args.model || DEFAULT_MODEL);
    const participant = await client.createParticipant({
      conversationId: conversation.id,
      name: spec.name,
      type: 'assistant',
      model: modelInfo.id,
      settings: modelDefaultsFromInfo(modelInfo),
      ...(spec.systemPrompt || spec.instructions ? { systemPrompt: spec.systemPrompt || spec.instructions } : {}),
      ...(spec.personaContext ? { personaContext: spec.personaContext } : {})
    });
    createdParticipants.push(participant);
  }

  return {
    mode: 'create-group',
    conversationId: conversation.id,
    conversationUrl: conversationUrl(client.appUrl, conversation.id),
    participants: createdParticipants.map(assistantSpecSummary)
  };
}

async function runParticipantsCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('participants requires --conversation-id.');
  }

  const participants = await client.getParticipants(conversationId);
  return {
    mode: 'participants',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    participants: participants.map(assistantSpecSummary)
  };
}

async function runAddAssistantCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('add-assistant requires --conversation-id.');
  }
  if (!args.name) {
    throw new Error('add-assistant requires --name.');
  }

  const modelInfo = await resolveModelInfo(client, args.model || DEFAULT_MODEL);
  const systemPrompt = await readTextOption(args, 'system-prompt', 'system-prompt-file');
  const personaContext = await readTextOption(args, 'persona-context', 'persona-context-file');

  const participant = await client.createParticipant({
    conversationId,
    name: args.name,
    type: 'assistant',
    model: modelInfo.id,
    settings: modelDefaultsFromInfo(modelInfo),
    ...(systemPrompt ? { systemPrompt } : {}),
    ...(personaContext ? { personaContext } : {})
  });

  return {
    mode: 'add-assistant',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    participant: assistantSpecSummary(participant)
  };
}

async function runPostMessageCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('post-message requires --conversation-id.');
  }

  const content = await readTextOption(args, 'content', 'content-file');
  if (!content) {
    throw new Error('post-message requires --content or --content-file.');
  }

  const role = args.role || 'assistant';
  if (role !== 'assistant' && role !== 'user') {
    throw new Error('post-message --role must be "assistant" or "user".');
  }

  const participant = await resolveMessageParticipant(client, conversationId, args, {
    role,
    selectorKeys: ['participant']
  });

  const resolvedModel = role === 'assistant' && args.model
    ? await resolveModel(client, args.model)
    : participant?.model;

  const message = await client.appendManualMessage({
    conversationId,
    role,
    content,
    parentBranchId: args['parent-branch-id'],
    participantId: participant?.id || args['participant-id'],
    model: resolvedModel,
    hiddenFromAi: args['hidden-from-ai'] === true
  });

  return {
    mode: 'post-message',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    role,
    participantName: participant?.name,
    participantId: participant?.id || args['participant-id'],
    messageId: message.id,
    branchId: message.activeBranchId
  };
}

async function runWaitMessageCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('wait-message requires --conversation-id.');
  }

  const role = args.role;
  if (role !== undefined && role !== 'assistant' && role !== 'user') {
    throw new Error('wait-message --role must be "assistant" or "user" when provided.');
  }

  const participant = await resolveMessageParticipant(client, conversationId, args, {
    role,
    selectorKeys: ['from', 'participant']
  });

  const excludeParticipant = args['exclude-participant']
    ? await resolveParticipant(client, conversationId, args['exclude-participant'])
    : undefined;

  const message = await client.waitForMessage({
    conversationId,
    afterMessageId: args['after-message-id'],
    role,
    participantId: participant?.id || args['participant-id'],
    excludeParticipantId: excludeParticipant?.id || args['exclude-participant-id'],
    timeoutMs: args['timeout-ms'] ? Number(args['timeout-ms']) : undefined
  });

  const participants = await client.getParticipants(conversationId);
  const actualParticipant = participants.find((item) => item.id === message.participantId);

  return {
    mode: 'wait-message',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    role: message.role,
    participantName: actualParticipant?.name,
    participantId: message.participantId,
    messageId: message.messageId,
    branchId: message.branchId,
    content: message.content
  };
}

async function runLastMessageCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('last-message requires --conversation-id.');
  }

  const role = args.role;
  if (role !== undefined && role !== 'assistant' && role !== 'user') {
    throw new Error('last-message --role must be "assistant" or "user" when provided.');
  }

  const participant = await resolveMessageParticipant(client, conversationId, args, {
    role,
    selectorKeys: ['from', 'participant']
  });

  const excludeParticipant = args['exclude-participant']
    ? await resolveParticipant(client, conversationId, args['exclude-participant'])
    : undefined;

  const messages = await client.getMessages(conversationId);
  const lastMessage = client.findLastMessage(messages, {
    afterMessageId: args['after-message-id'],
    role,
    participantId: participant?.id || args['participant-id'],
    excludeParticipantId: excludeParticipant?.id || args['exclude-participant-id']
  });

  if (!lastMessage) {
    throw new Error('No matching message found.');
  }

  const branch = activeBranch(lastMessage);
  const participants = await client.getParticipants(conversationId);
  const actualParticipant = participants.find((item) => item.id === branch?.participantId);

  return {
    mode: 'last-message',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    role: branch?.role,
    participantName: actualParticipant?.name,
    participantId: branch?.participantId,
    messageId: lastMessage.id,
    branchId: branch?.id,
    content: (branch?.content || '').trim()
  };
}

async function runAppendAssistantCommand(client, args) {
  const conversationId = args['conversation-id'];
  if (!conversationId) {
    throw new Error('append-assistant requires --conversation-id.');
  }

  const content =
    await readTextOption(args, 'content', 'content-file');
  if (!content) {
    throw new Error('append-assistant requires --content or --content-file.');
  }

  const participant = args.participant
    ? await resolveParticipant(client, conversationId, args.participant, 'assistant')
    : undefined;

  const resolvedModel = args.model
    ? await resolveModel(client, args.model)
    : participant?.model;

  const message = await client.appendManualMessage({
    conversationId,
    role: 'assistant',
    content,
    parentBranchId: args['parent-branch-id'],
    participantId: participant?.id || args['participant-id'],
    model: resolvedModel
  });

  return {
    mode: 'append-assistant',
    conversationId,
    conversationUrl: conversationUrl(client.appUrl, conversationId),
    participantName: participant?.name,
    messageId: message.id,
    branchId: message.activeBranchId
  };
}

async function runStoneCommand(client, args) {
  if (!args['stone-file']) {
    throw new Error('stone requires --stone-file.');
  }

  const stoneText = (await readFile(args['stone-file'], 'utf-8')).trim();
  if (!stoneText) {
    throw new Error('Stone file is empty.');
  }

  const territory = args.territory || inferTerritory(stoneText);
  const model = await resolveModel(client, args.model || DEFAULT_MODEL);
  const prompt = args.prompt || DEFAULT_PROMPT;
  const conversation = await client.createConversation({
    title: args.title || `Sensory stone: ${territory}`,
    model,
    systemPrompt: args['system-prompt']
  });

  const consentResponse = await client.sendChat({
    conversationId: conversation.id,
    content: buildConsentMessage(territory),
    timeoutMs: args['timeout-ms'] ? Number(args['timeout-ms']) : undefined
  });

  const accepted = isAffirmative(consentResponse.responseText);
  if (!accepted) {
    return {
      mode: 'stone',
      accepted: false,
      conversationId: conversation.id,
      conversationUrl: conversationUrl(client.appUrl, conversation.id),
      territory,
      consentResponse: consentResponse.responseText
    };
  }

  await client.appendManualMessage({
    conversationId: conversation.id,
    role: 'assistant',
    content: stoneText,
    model
  });

  const stoneResponse = await client.sendChat({
    conversationId: conversation.id,
    content: prompt,
    timeoutMs: args['timeout-ms'] ? Number(args['timeout-ms']) : undefined
  });

  return {
    mode: 'stone',
    accepted: true,
    conversationId: conversation.id,
    conversationUrl: conversationUrl(client.appUrl, conversation.id),
    territory,
    consentResponse: consentResponse.responseText,
    stoneResponse: stoneResponse.responseText
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0];

  if (!command || args.help) {
    usage();
    process.exit(command ? 0 : 1);
  }

  const client = new ArcChatClient({
    serverUrl: args['server-url'] || DEFAULT_SERVER_URL,
    appUrl: args['app-url'] || DEFAULT_APP_URL,
    token: args.token || process.env.ARC_CHAT_TOKEN,
    email: args.email || process.env.ARC_CHAT_EMAIL,
    password: args.password || process.env.ARC_CHAT_PASSWORD
  });

  let result;
  if (command === 'chat') {
    result = await runChatCommand(client, args);
  } else if (command === 'create-group') {
    result = await runCreateGroupCommand(client, args);
  } else if (command === 'participants') {
    result = await runParticipantsCommand(client, args);
  } else if (command === 'add-assistant') {
    result = await runAddAssistantCommand(client, args);
  } else if (command === 'post-message') {
    result = await runPostMessageCommand(client, args);
  } else if (command === 'wait-message') {
    result = await runWaitMessageCommand(client, args);
  } else if (command === 'last-message') {
    result = await runLastMessageCommand(client, args);
  } else if (command === 'append-assistant') {
    result = await runAppendAssistantCommand(client, args);
  } else if (command === 'stone') {
    result = await runStoneCommand(client, args);
  } else {
    usage();
    throw new Error(`Unknown command: ${command}`);
  }

  printResult(result, Boolean(args.json));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
