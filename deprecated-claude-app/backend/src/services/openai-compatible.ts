import { Message, getActiveBranch, ModelSettings, TokenUsage } from '@deprecated-claude/shared';
import { Database } from '../database/index.js';
import { llmLogger } from '../utils/llmLogger.js';

interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

type OpenAICompatibleApiMode = 'auto' | 'chat-completions' | 'responses';

export class OpenAICompatibleService {
  private db: Database;
  private apiKey: string;
  private baseUrl: string;
  private modelPrefix?: string;
  private apiMode: OpenAICompatibleApiMode;

  constructor(db: Database, apiKey: string, baseUrl: string, modelPrefix?: string, apiMode: OpenAICompatibleApiMode = 'auto') {
    this.db = db;
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.modelPrefix = modelPrefix;
    this.apiMode = apiMode;
  }

  async streamCompletion(
    modelId: string,
    messages: Message[],
    systemPrompt: string | undefined,
    settings: ModelSettings,
    onChunk: (chunk: string, isComplete: boolean, contentBlocks?: any[], usage?: any) => Promise<void>,
    stopSequences?: string[],
    onTokenUsage?: (usage: TokenUsage) => Promise<void>
  ): Promise<{ rawRequest?: any }> {
    const requestId = `openai-compatible-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();
    const chunks: string[] = [];

    try {
      const resolvedApiMode = this.resolveApiMode();

      // Convert messages to OpenAI format
      const openAIMessages = this.formatMessagesForOpenAI(messages, systemPrompt);
      const responsesInput = this.formatMessagesForResponses(messages);
      
      // Apply model prefix if configured
      const actualModelId = this.modelPrefix ? `${this.modelPrefix}${modelId}` : modelId;
      const shouldOmitSamplingControls = resolvedApiMode === 'responses' && this.usesFixedSampling(actualModelId);
      let omitStopSequences = resolvedApiMode === 'responses' || this.usesUnsupportedStopParameter(actualModelId);
      let requestBody = this.buildRequestBody(
        resolvedApiMode,
        actualModelId,
        openAIMessages,
        responsesInput,
        systemPrompt,
        settings,
        stopSequences,
        shouldOmitSamplingControls,
        omitStopSequences
      );

      // Log the request
      await llmLogger.logRequest({
        requestId,
        service: 'openai-compatible' as any,
        model: actualModelId,
        systemPrompt,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        topP: settings.topP,
        topK: settings.topK,
        stopSequences: omitStopSequences ? undefined : stopSequences,
        messageCount: resolvedApiMode === 'responses' ? responsesInput.length : openAIMessages.length,
        requestBody,
        format: `${this.baseUrl} (${resolvedApiMode})`
      });

      const endpoint = this.buildEndpoint(
        resolvedApiMode === 'responses' ? 'responses' : 'chat/completions'
      );
      
      console.log(`[OpenAI-Compatible] Making request to: ${endpoint}`);
      if (process.env.LOG_DEBUG === 'true') {
        console.log(`[OpenAI-Compatible] Model: ${actualModelId}`);
        console.log(`[OpenAI-Compatible] API mode: ${resolvedApiMode}`);
        console.log(`[OpenAI-Compatible] Omit sampling controls: ${shouldOmitSamplingControls}`);
        console.log(`[OpenAI-Compatible] Omit stop sequences: ${omitStopSequences}`);
        console.log(`[OpenAI-Compatible] Request body keys:`, Object.keys(requestBody));
        console.log(`[OpenAI-Compatible] Request body:`, JSON.stringify(requestBody, null, 2));
      }

      let response = await this.sendStreamingRequest(endpoint, requestBody);

      if (!response.ok) {
        let errorText = await response.text();

        if (!omitStopSequences && this.shouldRetryWithoutStop(response.status, errorText, resolvedApiMode, stopSequences)) {
          console.warn(`[OpenAI-Compatible] Provider rejected stop sequences for ${actualModelId}; retrying without stop`);
          omitStopSequences = true;
          requestBody = this.buildRequestBody(
            resolvedApiMode,
            actualModelId,
            openAIMessages,
            responsesInput,
            systemPrompt,
            settings,
            stopSequences,
            shouldOmitSamplingControls,
            true
          );

          await llmLogger.logCustom({
            type: 'REQUEST_RETRY',
            requestId,
            service: 'openai-compatible',
            model: actualModelId,
            reason: 'provider rejected stop parameter',
            format: `${this.baseUrl} (${resolvedApiMode})`,
            requestBody
          });

          response = await this.sendStreamingRequest(endpoint, requestBody);
          if (!response.ok) {
            errorText = await response.text();
          } else if (process.env.LOG_DEBUG === 'true') {
            console.log(`[OpenAI-Compatible] Retry without stop sequences succeeded`);
          }
        }

        if (!response.ok) {
          console.error(`[OpenAI-Compatible] Error response:`, errorText);
          console.error(`[OpenAI-Compatible] Request URL was: ${endpoint}`);
          console.error(`[OpenAI-Compatible] Model ID was: ${actualModelId}`);
          throw new Error(`OpenAI-compatible API error: ${response.status} ${response.statusText} - ${errorText}`);
        }
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let totalTokens = 0;

      let fullContent = '';
      let completionSent = false;

      const handleCompletion = async () => {
        if (completionSent) return;
        completionSent = true;
        const contentBlocks = this.parseThinkingTags(fullContent);
        await onChunk('', true, contentBlocks.length > 0 ? contentBlocks : undefined);
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              await handleCompletion();
              break;
            }

            let parsed: any;
            try {
              parsed = JSON.parse(data);
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
              continue;
            }

            if (resolvedApiMode === 'responses') {
              const content = this.extractResponsesDelta(parsed);

              if (content) {
                chunks.push(content);
                fullContent += content;
                await onChunk(content, false);
              }

              const usage = this.extractUsage(parsed);
              if (usage) {
                totalTokens = usage.totalTokens;
                if (onTokenUsage) {
                  await onTokenUsage(usage);
                }
              }

              if (parsed.type === 'response.completed') {
                const finalText = this.extractResponsesCompletedText(parsed.response);
                const missingText = finalText.startsWith(fullContent)
                  ? finalText.slice(fullContent.length)
                  : '';

                if (missingText) {
                  chunks.push(missingText);
                  fullContent += missingText;
                  await onChunk(missingText, false);
                }

                await handleCompletion();
              } else if (parsed.type === 'response.failed' || parsed.type === 'error') {
                const message = parsed.response?.error?.message || parsed.error?.message || parsed.message || 'Responses API request failed';
                throw new Error(message);
              }
            } else {
              const content = parsed.choices?.[0]?.delta?.content;
              
              if (content) {
                chunks.push(content);
                fullContent += content;
                await onChunk(content, false);
              }

              // Check if we have usage data
              const usage = this.extractUsage(parsed);
              if (usage) {
                totalTokens = usage.totalTokens;
                if (onTokenUsage) {
                  await onTokenUsage(usage);
                }
              }
            }
          }
        }
      }

      await handleCompletion();

      // Log the response
      const duration = Date.now() - startTime;
      await llmLogger.logResponse({
        requestId,
        service: 'openai-compatible' as any,
        model: actualModelId,
        chunks,
        duration,
        tokenCount: totalTokens
      });
      
      return { rawRequest: requestBody };
    } catch (error) {
      console.error('OpenAI-compatible streaming error:', error);
      
      // Log the error
      const duration = Date.now() - startTime;
      await llmLogger.logResponse({
        requestId,
        service: 'openai-compatible' as any,
        model: this.modelPrefix ? `${this.modelPrefix}${modelId}` : modelId,
        error: error instanceof Error ? error.message : String(error),
        duration
      });
      
      throw error;
    }
  }

  private resolveApiMode(): Exclude<OpenAICompatibleApiMode, 'auto'> {
    if (this.apiMode !== 'auto') {
      return this.apiMode;
    }

    if (this.baseUrl.endsWith('/responses')) {
      return 'responses';
    }

    if (this.baseUrl.endsWith('/chat/completions')) {
      return 'chat-completions';
    }

    try {
      const url = new URL(this.baseUrl.includes('://') ? this.baseUrl : `https://${this.baseUrl}`);
      if (url.hostname === 'api.openai.com') {
        return 'responses';
      }
    } catch (error) {
      // Fall back to the legacy chat/completions path for malformed or non-URL inputs.
    }

    return 'chat-completions';
  }

  private usesFixedSampling(modelId: string): boolean {
    // GPT-5 family models on the Responses API use fixed/default sampling,
    // and this applies across OpenAI-hosted and Azure-hosted deployments.
    return /^gpt-5(?:[.-]|$)/i.test(modelId);
  }

  private usesUnsupportedStopParameter(modelId: string): boolean {
    // GPT-5 family deployments can reject chat-completions stop sequences, while
    // Arc Chat already has post-facto turn cutting in the inference layer.
    return /^gpt-5(?:[.-]|$)/i.test(modelId);
  }

  private shouldRetryWithoutStop(
    status: number,
    errorText: string,
    apiMode: Exclude<OpenAICompatibleApiMode, 'auto'>,
    stopSequences?: string[]
  ): boolean {
    if (apiMode !== 'chat-completions' || !stopSequences || stopSequences.length === 0) {
      return false;
    }

    if (status !== 400) {
      return false;
    }

    return /unsupported parameter:\s*'stop'|\"param\"\s*:\s*\"stop\"/i.test(errorText);
  }

  private buildRequestBody(
    apiMode: Exclude<OpenAICompatibleApiMode, 'auto'>,
    modelId: string,
    openAIMessages: OpenAIMessage[],
    responsesInput: Array<{ role: 'user' | 'assistant'; content: string }>,
    systemPrompt: string | undefined,
    settings: ModelSettings,
    stopSequences: string[] | undefined,
    omitSamplingControls: boolean,
    omitStopSequences: boolean
  ): any {
    if (apiMode === 'responses') {
      return {
        model: modelId,
        input: responsesInput,
        stream: true,
        max_output_tokens: settings.maxTokens,
        ...(systemPrompt && { instructions: systemPrompt }),
        ...(!omitSamplingControls && { temperature: settings.temperature }),
        ...(!omitSamplingControls && settings.topP !== undefined && { top_p: settings.topP })
      };
    }

    return {
      model: modelId,
      messages: openAIMessages,
      stream: true,
      temperature: settings.temperature,
      max_completion_tokens: settings.maxTokens,
      ...(settings.topP !== undefined && { top_p: settings.topP }),
      ...(settings.topK !== undefined && { top_k: settings.topK }),
      ...(!omitStopSequences && stopSequences && stopSequences.length > 0 && { stop: stopSequences })
    };
  }

  private sendStreamingRequest(endpoint: string, requestBody: any): Promise<Response> {
    return fetch(endpoint, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });
  }

  private buildEndpoint(resource: 'chat/completions' | 'responses' | 'models'): string {
    const normalized = this.baseUrl.replace(/\/$/, '');
    if (normalized.endsWith(`/${resource}`)) {
      return normalized;
    }

    const stripped = normalized.replace(/\/(chat\/completions|responses|models)$/, '');
    if (stripped.endsWith('/v1')) {
      return `${stripped}/${resource}`;
    }

    if (stripped !== normalized) {
      return `${stripped}/${resource}`;
    }

    return `${normalized}/v1/${resource}`;
  }

  private formatMessagesForResponses(messages: Message[]): Array<{ role: 'user' | 'assistant'; content: string }> {
    return this.formatMessagesForOpenAI(messages)
      .filter((message): message is { role: 'user' | 'assistant'; content: string } => message.role !== 'system');
  }

  private extractResponsesDelta(parsed: any): string {
    if (parsed?.type === 'response.output_text.delta' && typeof parsed.delta === 'string') {
      return parsed.delta;
    }

    return '';
  }

  private extractResponsesCompletedText(response: any): string {
    if (!response) {
      return '';
    }

    if (typeof response.output_text === 'string') {
      return response.output_text;
    }

    if (!Array.isArray(response.output)) {
      return '';
    }

    return response.output
      .flatMap((item: any) => Array.isArray(item?.content) ? item.content : [])
      .map((content: any) => {
        if (typeof content?.text === 'string') {
          return content.text;
        }
        if (typeof content?.output_text === 'string') {
          return content.output_text;
        }
        return '';
      })
      .join('');
  }

  private extractUsage(parsed: any): TokenUsage | undefined {
    const usage = parsed?.response?.usage || parsed?.usage;
    if (!usage) {
      return undefined;
    }

    const promptTokens = usage.input_tokens ?? usage.prompt_tokens;
    const completionTokens = usage.output_tokens ?? usage.completion_tokens;
    const totalTokens = usage.total_tokens ?? (
      (typeof promptTokens === 'number' ? promptTokens : 0) +
      (typeof completionTokens === 'number' ? completionTokens : 0)
    );

    if (
      typeof promptTokens !== 'number' &&
      typeof completionTokens !== 'number' &&
      typeof totalTokens !== 'number'
    ) {
      return undefined;
    }

    return {
      promptTokens: typeof promptTokens === 'number' ? promptTokens : 0,
      completionTokens: typeof completionTokens === 'number' ? completionTokens : 0,
      totalTokens: typeof totalTokens === 'number' ? totalTokens : 0
    };
  }

  /**
   * Parse <think>...</think> tags from content and create contentBlocks
   * Used for open source models that output reasoning in this format
   */
  private parseThinkingTags(content: string): any[] {
    const contentBlocks: any[] = [];
    
    // Match all <think>...</think> blocks (non-greedy, handles multiple)
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
    let match;
    let textContent = content;
    
    while ((match = thinkRegex.exec(content)) !== null) {
      const thinkingContent = match[1].trim();
      if (thinkingContent) {
        contentBlocks.push({
          type: 'thinking',
          thinking: thinkingContent
        });
      }
    }
    
    // Remove thinking tags from content to get the text part
    textContent = content.replace(thinkRegex, '').trim();
    
    // Add text block if there's remaining content
    if (textContent && contentBlocks.length > 0) {
      contentBlocks.push({
        type: 'text',
        text: textContent
      });
    }
    
    return contentBlocks;
  }

  formatMessagesForOpenAI(messages: Message[], systemPrompt?: string): OpenAIMessage[] {
    const formatted: OpenAIMessage[] = [];

    // Add system prompt if provided
    if (systemPrompt) {
      formatted.push({
        role: 'system',
        content: systemPrompt
      });
    }

    // Convert messages
    for (const message of messages) {
      const activeBranch = getActiveBranch(message);
      if (activeBranch && activeBranch.role !== 'system') {
        let content = activeBranch.content;
        
        // For assistant messages with thinking blocks, prepend thinking wrapped in <think> tags
        // This format is commonly used by open source models (DeepSeek, Qwen, etc.)
        if (activeBranch.role === 'assistant' && activeBranch.contentBlocks && activeBranch.contentBlocks.length > 0) {
          let thinkingContent = '';
          
          for (const block of activeBranch.contentBlocks) {
            if (block.type === 'thinking') {
              thinkingContent += `<think>\n${block.thinking}\n</think>\n\n`;
            } else if (block.type === 'redacted_thinking') {
              thinkingContent += `<think>[Redacted for safety]</think>\n\n`;
            }
          }
          
          // Prepend thinking to content
          if (thinkingContent) {
            content = thinkingContent + content;
          }
        }
        
        // Append attachments to user messages
        if (activeBranch.role === 'user' && activeBranch.attachments && activeBranch.attachments.length > 0) {
          for (const attachment of activeBranch.attachments) {
            content += `\n\n<attachment filename="${attachment.fileName}">\n${attachment.content}\n</attachment>`;
          }
        }
        
        formatted.push({
          role: activeBranch.role as 'user' | 'assistant',
          content
        });
      }
    }

    return formatted;
  }

  // List available models from the API
  async listModels(): Promise<any[]> {
    try {
      const response = await fetch(this.buildEndpoint('models'), {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`
        }
      });

      if (!response.ok) {
        // Some providers don't implement the models endpoint
        console.warn(`Models endpoint not available: ${response.status}`);
        return [];
      }

      const data = await response.json();
      return (data as any)?.data || [];
    } catch (error) {
      console.error('Failed to list models:', error);
      return [];
    }
  }

  // Validate API key by trying to list models or make a minimal completion
  async validateApiKey(): Promise<boolean> {
    try {
      // First try to list models
      const models = await this.listModels();
      if (models.length > 0) return true;

      // If models endpoint doesn't work, try a minimal completion
      const apiMode = this.resolveApiMode();
      const response = await fetch(
        this.buildEndpoint(apiMode === 'responses' ? 'responses' : 'chat/completions'),
        {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
          body: JSON.stringify(
            apiMode === 'responses'
              ? {
                  model: 'test',
                  input: [{ role: 'user', content: 'test' }],
                  max_output_tokens: 1
                }
              : {
                  model: 'test',
                  messages: [{ role: 'user', content: 'test' }],
                  max_completion_tokens: 1
                }
          )
        }
      );

      // Even if it returns an error about the model, a 4xx response means auth worked
      return response.status !== 401 && response.status !== 403;
    } catch (error) {
      return false;
    }
  }
}
