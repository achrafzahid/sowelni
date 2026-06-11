import {
  AfterViewChecked, Component, ElementRef,
  OnDestroy, OnInit, computed, effect, inject, viewChild,
} from '@angular/core';
import { AudioRecordingService } from '../../services/audio-recording';
import { WebsocketService } from '../../services/websocket';

@Component({
  selector: 'app-transcriber',
  standalone: true,
  templateUrl: './transcriber.html',
  styleUrl: './transcriber.scss',
})
export class TranscriberComponent implements OnInit, OnDestroy, AfterViewChecked {
  private readonly ws = inject(WebsocketService);
  private readonly recorder = inject(AudioRecordingService);

  readonly conversationEl = viewChild<ElementRef<HTMLElement>>('conversation');

  readonly connected = this.ws.connected;
  readonly recording = this.recorder.recording;
  readonly turns = this.ws.turns;
  readonly currentTranscript = this.ws.currentTranscript;
  readonly currentAnswer = this.ws.currentAnswer;
  readonly isStreamingAnswer = this.ws.isStreamingAnswer;
  readonly lastError = this.ws.lastError;
  readonly micError = this.recorder.micError;
  readonly hasContent = this.ws.hasContent;
  readonly levels = this.recorder.levels;

  readonly statusLabel = computed(() => {
    if (!this.connected()) return 'Offline';
    if (this.recording()) return 'Listening';
    if (this.isStreamingAnswer()) return 'Thinking';
    return 'Ready';
  });

  readonly statusKind = computed<'idle' | 'live' | 'busy' | 'off'>(() => {
    if (!this.connected()) return 'off';
    if (this.recording()) return 'live';
    if (this.isStreamingAnswer()) return 'busy';
    return 'idle';
  });

  private shouldScroll = false;

  constructor() {
    effect(() => {
      this.turns();
      this.currentTranscript();
      this.currentAnswer();
      this.shouldScroll = true;
    });
  }

  ngOnInit(): void { this.ws.connect(); }
  ngOnDestroy(): void { this.recorder.stop(); this.ws.disconnect(); }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.shouldScroll = false;
      const el = this.conversationEl()?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    }
  }

  async toggle(): Promise<void> {
    if (this.recording()) {
      this.recorder.stop();
      setTimeout(() => this.ws.requestAnswer(), 450);
    } else {
      try { await this.recorder.start(); } catch { /* micError set */ }
    }
  }

  clearHistory(): void {
    if (this.recording()) this.recorder.stop();
    this.ws.clearConversation();
  }

  trackTurn(_: number, turn: { id: string }): string { return turn.id; }
}
