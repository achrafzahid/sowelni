import { Injectable, NgZone, inject, signal } from '@angular/core';
import { WebsocketService } from './websocket';

const CHUNK_MS = 2000;
const BAR_COUNT = 24;
const MIME_PREFS = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];

@Injectable({ providedIn: 'root' })
export class AudioRecordingService {
  private readonly ws = inject(WebsocketService);
  private readonly zone = inject(NgZone);

  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private active = false;
  private mime = 'audio/webm';

  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private raf: number | null = null;
  private freq: Uint8Array | null = null;

  readonly recording = signal(false);
  readonly micError = signal<string | null>(null);
  readonly levels = signal<number[]>(new Array(BAR_COUNT).fill(0));

  async start(): Promise<void> {
    if (this.active) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      this.mime = MIME_PREFS.find((m) => MediaRecorder.isTypeSupported(m)) || 'audio/webm';
      this.active = true;
      this.recording.set(true);
      this.micError.set(null);
      this.setupAnalyser();
      this.recordSegment();
    } catch (err) {
      this.micError.set(err instanceof Error ? err.message : 'Mic access denied');
      this.cleanup();
      throw err;
    }
  }

  stop(): void {
    this.active = false;
    this.recording.set(false);
    if (this.timer !== null) { clearTimeout(this.timer); this.timer = null; }
    if (this.recorder?.state === 'recording') this.recorder.stop();
    this.teardownAnalyser();
    this.cleanup();
    this.levels.set(new Array(BAR_COUNT).fill(0));
  }

  // ── segment chaining ────────────────────────────────

  private recordSegment(): void {
    if (!this.active || !this.stream) return;
    this.chunks = [];

    const rec = new MediaRecorder(this.stream, { mimeType: this.mime });

    rec.ondataavailable = (e: BlobEvent) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };

    rec.onstop = async () => {
      const blob = new Blob(this.chunks, { type: this.mime });
      this.chunks = [];
      if (blob.size > 0) {
        this.ws.sendAudioChunk(await blob.arrayBuffer());
      }
      if (this.active) this.recordSegment();
    };

    rec.start();
    this.recorder = rec;
    this.timer = setTimeout(() => {
      if (this.recorder?.state === 'recording') this.recorder.stop();
    }, CHUNK_MS);
  }

  // ── live waveform ───────────────────────────────────

  private setupAnalyser(): void {
    if (!this.stream) return;
    const AC = window.AudioContext || (window as any).webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 128;
    this.analyser.smoothingTimeConstant = 0.7;
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.source.connect(this.analyser);
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
    this.zone.runOutsideAngular(() => this.tick());
  }

  private tick = (): void => {
    if (!this.analyser || !this.freq || !this.active) return;
    this.analyser.getByteFrequencyData(this.freq);
    const bins = this.freq.length;
    const group = Math.floor(bins / BAR_COUNT) || 1;
    const bars: number[] = [];
    for (let i = 0; i < BAR_COUNT; i++) {
      let sum = 0;
      const s = i * group, e = Math.min(s + group, bins);
      for (let j = s; j < e; j++) sum += this.freq[j];
      bars.push(Math.min(1, Math.pow(sum / (e - s) / 255, 0.7)));
    }
    this.zone.run(() => this.levels.set(bars));
    this.raf = requestAnimationFrame(this.tick);
  };

  private teardownAnalyser(): void {
    if (this.raf !== null) { cancelAnimationFrame(this.raf); this.raf = null; }
    this.source?.disconnect();
    this.source = null;
    this.analyser = null;
    this.freq = null;
    if (this.ctx?.state !== 'closed') this.ctx?.close().catch(() => {});
    this.ctx = null;
  }

  private cleanup(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
  }
}
