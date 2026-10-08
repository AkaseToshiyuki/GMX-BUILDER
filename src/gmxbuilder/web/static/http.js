/* JSON reads recover from one transient transport failure; writes are never replayed. */
(function () {
  'use strict';
  const readOnlyPosts = /^\/api\/(forcefield-compatibility|ligand-charge-suggestions|ligand-chemistry)\/[a-f0-9]+$/;
  const transientStatus = new Set([408, 502, 503, 504]);

  async function requestJson(url, options = {}, contract = {}) {
    const method = String(options.method || 'GET').toUpperCase();
    const canRetry = method === 'GET' || (method === 'POST' && readOnlyPosts.test(String(url)));
    const current = () => !options.signal?.aborted && (!contract.current || contract.current());
    for (let attempt = 0; ; attempt++) {
      if (!current()) throw new DOMException('Superseded request', 'AbortError');
      let response, payload, failure, transient = false;
      try {
        response = await fetch(url, options);
        try { payload = await response.json(); }
        catch (error) {
          if (error.name === 'AbortError') throw error;
          failure = new Error('The server returned an empty or invalid response. Please retry.');
          transient = response.ok || transientStatus.has(response.status);
        }
        if (!response.ok) {
          const detail = payload?.error || payload?.detail;
          failure = new Error(typeof detail === 'string' ? detail :
            'Request failed (HTTP ' + response.status + '). Please retry.');
          transient = transientStatus.has(response.status);
        } else if (!failure) {
          if (payload?.error && !contract.allowErrorPayload) failure = new Error(String(payload.error));
          else if (payload === null || typeof payload !== 'object' ||
                   (contract.validate && !contract.validate(payload))) {
            failure = new Error('The server returned incomplete data. Please retry.');
            transient = true;
          }
        }
      } catch (error) {
        if (error.name === 'AbortError') throw error;
        failure = new Error('The server could not complete the request. Please retry.');
        transient = error instanceof TypeError || error instanceof SyntaxError;
      }
      if (!current()) throw new DOMException('Superseded request', 'AbortError');
      if (!failure) return payload;
      if (!canRetry || !transient || attempt >= 1) throw failure;
      await new Promise(resolve => setTimeout(resolve, 300));
    }
  }
  async function json(url, options = {}, contract = {}) {
    const method = String(options.method || 'GET').toUpperCase();
    const read = method === 'GET' || (method === 'POST' && readOnlyPosts.test(String(url)));
    const timeoutMs = contract.timeoutMs ?? (read ? 15000 : 0);
    if (!timeoutMs) return requestJson(url, options, contract);
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (options.signal?.aborted) controller.abort();
    options.signal?.addEventListener('abort', abort, {once:true});
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
    try { return await requestJson(url, {...options, signal:controller.signal}, contract); }
    catch (error) {
      if (timedOut && !options.signal?.aborted)
        throw new Error('The request timed out. Retry this section; running tasks are unchanged.');
      throw error;
    } finally {
      clearTimeout(timer); options.signal?.removeEventListener('abort', abort);
    }
  }
  window.GMXHttp = { json };
  const reads = new Map();
  // Concurrent widgets observing the same URL share one bounded read.
  window.GMXPoll = {
    read(url) {
      if (!reads.has(url)) reads.set(url, json(url, {}, {allowErrorPayload:true}).finally(() => reads.delete(url)));
      return reads.get(url);
    },
    start(fn, interval, immediate = false) {
      const handle = {stopped:false, timer:null};
      const tick = async () => {
        if (handle.stopped) return;
        try { if (!document.hidden) await fn(); } catch (_error) { /* Next read can recover. */ }
        if (!handle.stopped) handle.timer = setTimeout(tick, document.hidden ? Math.max(5000, interval) : interval);
      };
      handle.stop = () => { handle.stopped = true; clearTimeout(handle.timer); };
      handle.timer = setTimeout(tick, immediate ? 0 : interval);
      return handle;
    },
    stop(handle) { if (handle?.stop) handle.stop(); else clearInterval(handle); }
  };

})();

window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('http.js');
