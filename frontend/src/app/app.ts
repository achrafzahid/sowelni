import { Component } from '@angular/core';
import { TranscriberComponent } from './components/transcriber/transcriber';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [TranscriberComponent],
  template: '<app-transcriber />',
})
export class App {}
