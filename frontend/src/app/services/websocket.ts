import { Injectable, computed, signal } from '@angular/core';
import { io, type Socket } from 'socket.io-client';
import { environment } from '../../environments/environment';

export interface ConversationTurn {
  id: string;
  question: string;
  answer: string;
  timestamp: number;
}

@Injectable({ providedIn: 'root' })
export class WebsocketService {
  private socket: Socket | null = null;

  readonly connected = signal(false);
  readonly lastError = signal<string | null>(null);
  readonly currentTranscript = signal('');
  readonly currentAnswer = signal('');
  readonly isStreamingAnswer = signal(false);
  readonly turns = signal<ConversationTurn[]>([]);

  readonly hasContent = computed(
    () =>
      this.turns().length > 0 ||
      this.currentTranscript().length > 0 ||
      this.currentAnswer().length > 0,
  );

  connect(): void {
    if (this.socket?.connected) return;

    const s = io(environment.socketUrl, {
      transports: ['websocket'],
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 500,
    });

    s.on('connect', () => {
      this.connected.set(true);
      this.lastError.set(null);
    });

    s.on('disconnect', () => this.connected.set(false));

    s.on('connect_error', (err: Error) => {
      this.lastError.set('Connection failed: ' + err.message);
    });

    s.on('transcription', (data: { partial: string; full: string }) => {
      this.currentTranscript.set(data.full);
    });

    s.on('llm_start', () => {
      this.isStreamingAnswer.set(true);
      this.currentAnswer.set('');
    });

    s.on('llm_token', (data: { token: string }) => {
      this.currentAnswer.update((prev) => prev + data.token);
    });

    s.on('llm_done', (data: { answer: string; question: string }) => {
      this.isStreamingAnswer.set(false);
      if (data.question || data.answer) {
        const turn: ConversationTurn = {
          id: Date.now() + '-' + Math.random().toString(36).slice(2, 8),
          question: data.question,
          answer: data.answer,
          timestamp: Date.now(),
        };
        this.turns.update((prev) => [...prev, turn]);
      }
      this.currentTranscript.set('');
      this.currentAnswer.set('');
    });

    s.on('error', (data: { message: string }) => {
      this.lastError.set(data.message);
    });

    this.socket = s;
  }

  sendAudioChunk(buffer: ArrayBuffer): void {
    this.socket?.emit('audio_chunk', buffer);
  }

  requestAnswer(): void {
    this.socket?.emit('request_answer');
  }

  clearConversation(): void {
    this.turns.set([]);
    this.currentTranscript.set('');
    this.currentAnswer.set('');
    this.lastError.set(null);
    this.socket?.emit('reset');
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
    this.connected.set(false);
  }
}
