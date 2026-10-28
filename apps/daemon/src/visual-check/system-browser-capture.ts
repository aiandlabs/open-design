// Headless capture through the system Chrome / Edge / Chromium the daemon
// already knows how to launch (`browser-sessions.ts`, which honours
// OD_BROWSER_EXECUTABLE_PATH). This is the fallback renderer for visual
// check when the desktop app's bundled Chromium is not reachable — for
// example under `tools-dev run web`, `od daemon start --headless`, or a
// server deployment. It speaks the Chrome DevTools Protocol over Node's
// built-in WebSocket, so it adds no dependency.

import type { BrowserSessionService } from '../browser-sessions.js';

export interface SystemBrowserCaptureInput {
  url: string;
  width: number;
  height: number;
  mobile: boolean;
  /** Full-page captures stop at this many CSS pixels. */
  maxHeight?: number;
  timeoutMs?: number;
}

type Pending = { resolve: (value: any) => void; reject: (error: Error) => void };

class CdpConnection {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private readonly listeners = new Set<(method: string, params: any, sessionId?: string) => void>();

  private constructor(private readonly ws: WebSocket) {
    ws.addEventListener('message', (event) => {
      let msg: any;
      try {
        msg = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (typeof msg.id === 'number') {
        const waiter = this.pending.get(msg.id);
        if (!waiter) return;
        this.pending.delete(msg.id);
        if (msg.error) waiter.reject(new Error(`CDP ${msg.error.message ?? 'error'}`));
        else waiter.resolve(msg.result ?? {});
        return;
      }
      if (typeof msg.method === 'string') {
        for (const listener of this.listeners) listener(msg.method, msg.params, msg.sessionId);
      }
    });
    ws.addEventListener('close', () => {
      for (const waiter of this.pending.values()) waiter.reject(new Error('CDP connection closed'));
      this.pending.clear();
    });
  }

  static open(url: string, timeoutMs: number): Promise<CdpConnection> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      const timer = setTimeout(() => {
        ws.close();
        reject(new Error('CDP connection timed out'));
      }, timeoutMs);
      ws.addEventListener('open', () => {
        clearTimeout(timer);
        resolve(new CdpConnection(ws));
      }, { once: true });
      ws.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('CDP connection failed'));
      }, { once: true });
    });
  }

  send(method: string, params: Record<string, unknown> = {}, sessionId?: string): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }

  waitForEvent(method: string, sessionId: string, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.listeners.delete(listener);
        reject(new Error(`timed out waiting for ${method}`));
      }, timeoutMs);
      const listener = (m: string, _params: unknown, sid?: string) => {
        if (m !== method || sid !== sessionId) return;
        clearTimeout(timer);
        this.listeners.delete(listener);
        resolve();
      };
      this.listeners.add(listener);
    });
  }

  close(): void {
    try {
      this.ws.close();
    } catch {
      // already closed
    }
  }
}

export async function captureWithSystemBrowser(
  browserSessions: Pick<BrowserSessionService, 'create' | 'close'>,
  input: SystemBrowserCaptureInput,
): Promise<Buffer> {
  const timeoutMs = input.timeoutMs ?? 45_000;
  const maxHeight = input.maxHeight ?? 4000;
  const session = await browserSessions.create();
  let cdp: CdpConnection | null = null;
  try {
    cdp = await CdpConnection.open(session.websocketUrl, 10_000);
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: input.width,
      height: input.height,
      deviceScaleFactor: 1,
      mobile: input.mobile,
    }, sessionId);
    if (input.mobile) {
      await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true }, sessionId);
    }
    const loaded = cdp.waitForEvent('Page.loadEventFired', sessionId, timeoutMs);
    const nav = await cdp.send('Page.navigate', { url: input.url }, sessionId);
    if (nav?.errorText) throw new Error(`navigation failed: ${nav.errorText}`);
    await loaded;
    // Let web fonts and one animation frame settle before measuring.
    await cdp.send('Runtime.evaluate', {
      expression: 'Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise((r) => setTimeout(r, 3000))]).then(() => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 250))))',
      awaitPromise: true,
    }, sessionId);
    const measured = await cdp.send('Runtime.evaluate', {
      expression: 'Math.max(document.documentElement ? document.documentElement.scrollHeight : 0, document.body ? document.body.scrollHeight : 0)',
      returnByValue: true,
    }, sessionId);
    const docHeight = Number(measured?.result?.value) || input.height;
    const height = Math.min(maxHeight, Math.max(input.height, Math.ceil(docHeight)));
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: true,
      clip: { x: 0, y: 0, width: input.width, height, scale: 1 },
    }, sessionId);
    if (typeof shot?.data !== 'string' || !shot.data) throw new Error('browser returned no screenshot');
    return Buffer.from(shot.data, 'base64');
  } finally {
    cdp?.close();
    await browserSessions.close(session.id).catch(() => false);
  }
}
