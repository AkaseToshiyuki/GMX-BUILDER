/* One final-response contract for fetch and progress-reporting XHR uploads. */
(function () {
  'use strict';
  var nativeFetch = window.fetch.bind(window);
  var watched = new Map();

  function remember(ticket) {
    try { sessionStorage.setItem('gmxbuilder-operation-' + ticket.operation_id, JSON.stringify(ticket)); }
    catch (_error) { /* Explicit Task ID recovery remains available. */ }
  }
  function forget(id) {
    try { sessionStorage.removeItem('gmxbuilder-operation-' + id); } catch (_error) {}
    watched.delete(id);
  }
  function update(ticket) {
    remember(ticket);
    if (typeof showComputeQueueStatus === 'function') showComputeQueueStatus(ticket);
  }
  async function awaitResult(ticket, signal) {
    var id = ticket.operation_id;
    update(ticket);
    while (true) {
      if (signal && signal.aborted) throw new DOMException('Request aborted', 'AbortError');
      var response = await nativeFetch('/api/operations/' + id, {signal:signal});
      if (!response.ok) {
        var failure = await response.clone().json().catch(function () { return {}; });
        update(Object.assign({}, ticket, {status:'failed',
          error:failure.error || 'Operation status is unavailable (HTTP ' + response.status + '). Resume the Task ID to retry.'}));
        forget(id);
        return response;
      }
      ticket = Object.assign({}, ticket, await response.json());
      update(ticket);
      if (ticket.response_ready || ['failed', 'cancelled'].includes(ticket.status)) {
        var result = await nativeFetch('/api/operations/' + id + '/result', {signal:signal});
        if (result.status !== 202) {
          var error = null;
          if (!result.ok) {
            var detail = await result.clone().json().catch(function () { return {}; });
            error = detail.error || detail.detail || 'Operation failed (HTTP ' + result.status + '). Review the step below.';
            if (typeof error !== 'string') error = JSON.stringify(error);
          }
          update(Object.assign({}, ticket, {status:result.ok ? 'completed' : 'failed', error:error}));
          forget(id);
          return result;
        }
      }
      // Poll once per second to balance UI feedback against status-request load.
      await new Promise(function (resolve) { setTimeout(resolve, 1000); });
    }
  }

  window.resolveManagedResponse = async function (response, options) {
    options = options || {};
    var id = response.headers.get('X-GMXBUILDER-Operation');
    if (response.status !== 202 || !id) return response;
    var ticket = await response.json();
    if (!/^[a-f0-9]{32}$/.test(id) || ticket.operation_id !== id) throw new Error('Invalid operation receipt');
    ticket.request_url = String(options.requestUrl || '');
    remember(ticket);
    if (ticket.request_url === '/api/upload-pdb' && typeof acceptedManagedUpload === 'function') {
      acceptedManagedUpload(ticket);
    }
    if (!watched.has(id)) watched.set(id, awaitResult(ticket, options.signal));
    try { return (await watched.get(id)).clone(); }
    catch (error) {
      // A cancelled preview may be retried while the same ticket completes.
      // Do not cache its rejected promise as the permanent result of that ID.
      forget(id);
      throw error;
    }
  };
  window.fetch = async function (input, options) {
    return window.resolveManagedResponse(await nativeFetch(input, options), {
      requestUrl: typeof input === 'string' ? input : input.url,
      signal: (options && options.signal) || (input && input.signal)
    });
  };

  document.addEventListener('DOMContentLoaded', function () {
    nativeFetch('/api/resource-policy').then(function (response) {
      return response.ok ? response.json() : null;
    }).then(function (policy) {
      var notice = document.getElementById('task-retention-notice');
      if (!notice || !policy) return;
      notice.textContent = 'Tasks expire ' + policy.task_lifetime_hours +
        ' hours after creation, including queue time. Viewing or downloading does not extend this time.';
    }).catch(function () {});
    // Refresh reconstructs authoritative task state through resumeTask. These
    // watchers retain progress until its queued resume can return that state;
    // they never replay a completed Check or turn a ticket into structure data.
    var current = null;
    try { current = JSON.parse(sessionStorage.getItem('gmxbuilder-current-task')); } catch (_error) {}
    if (!current || window.location.pathname === '/') return;
    try {
      Object.keys(sessionStorage).filter(function (key) {
        return key.startsWith('gmxbuilder-operation-');
      }).forEach(function (key) {
        var ticket;
        try { ticket = JSON.parse(sessionStorage.getItem(key)); } catch (_error) { return; }
        if (!ticket || ticket.task_id !== current.task_id || !/^[a-f0-9]{32}$/.test(ticket.operation_id)) return;
        watched.set(ticket.operation_id, awaitResult(ticket).catch(function () {
          watched.delete(ticket.operation_id);
          return new Response(JSON.stringify({error:'Connection lost. Resume the Task ID to reconnect.'}),
            {status:503, headers:{'Content-Type':'application/json'}});
        }));
      });
    } catch (_error) {}
  });
}());
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('resource_queue.js');
