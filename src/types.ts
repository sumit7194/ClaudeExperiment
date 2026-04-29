export type TranscriptSource = 'claude_desktop' | 'claude_code' | 'manual';
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export interface Transcript {
  id: string;
  title: string;
  source: TranscriptSource;
  sourceChatId?: string;
  createdAt: string;
  updatedAt: string;
  capturedAt: string;
  model?: string;
  messageCount: number;
  tokenEstimate?: number;
  summary?: string;
  tags: string[];
}

export interface Message {
  id: string;
  transcriptId: string;
  seq: number;
  role: MessageRole;
  content: string;
  contentJson?: unknown;
  createdAt?: string;
  metadata?: Record<string, unknown>;
}

export interface SaveTranscriptInput {
  source: TranscriptSource;
  sourceChatId?: string;
  title?: string;
  model?: string;
  messages: Array<{
    role: MessageRole;
    content: string;
    contentJson?: unknown;
    createdAt?: string;
    metadata?: Record<string, unknown>;
  }>;
  tags?: string[];
}
