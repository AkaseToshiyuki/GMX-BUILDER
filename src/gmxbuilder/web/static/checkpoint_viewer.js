/* Exact-checkpoint viewers share loading, component identity and readiness. */
(function () {
  'use strict';
  const scriptVersion = new URL(document.currentScript.src).searchParams.get('v') || '2';
  const loads = new Map();
  const scenes = new Map();
  const frame = () => new Promise((resolve) => requestAnimationFrame(resolve));
  let epoch = 0;
  const metrics = new Map();
  const MAX_DECODED_BYTES = 64 * 1048576;
  function releaseScene(id) {
    const scene = scenes.get(id);
    if (!scene) return;
    scenes.delete(id);
    scene.viewer.clear();
    const host = document.getElementById(id), canvas = host?.querySelector('canvas');
    const gl = canvas?.getContext('webgl2') || canvas?.getContext('webgl');
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
    host?.replaceChildren();
  }
  function atomAt(payload, index, local, component) {
    const text = key => payload.dictionaries[key].labels[payload.dictionaries[key].codes[index]];
    const residue = text('resids');
    const bonds = Array.from(payload.targets.subarray(payload.offsets[index], payload.offsets[index+1]));
    const atom = {index:local, x:payload.xyz[index*3],y:payload.xyz[index*3+1],z:payload.xyz[index*3+2],
      serial:payload.ids[index]+1,atom:text('names'),resn:text('resnames'),chain:text('chains'),
      resi:Number.isSafeInteger(Number(residue))?Number(residue):payload.dictionaries.resids.codes[index]+1,
      originalResi:residue,elem:text('elements'),bonds,bondOrder:bonds.map(()=>1),
      hetflag:!['PROTEIN','NUCLEIC_ACID'].includes(component.kind)};
    atom.style = GMXStyle.atomStyle(atom, component.kind, payload.data.resolution === 'coarse-grained');
    return atom;
  }
  function status(host, message, failed) {
    let el = document.getElementById(host.id + '-loading');
    if (!el) {
      el = document.createElement('div');
      el.id = host.id + '-loading';
      el.className = 'viewer-loading';
      el.setAttribute('role', 'status');
      host.before(el);
    }
    el.textContent = message;
    el.dataset.state = failed ? 'error' : 'loading';
  }
  async function load(step, host, active) {
    const task = state.taskId,
      key = task + '/' + step;
    const listener = { host, active };
    if (loads.has(key)) {
      loads.get(key).listeners.push(listener);
      if (active()) status(host, 'Loading the checked structure…');
      return loads.get(key).promise;
    }
    const controller = new AbortController();
    const item = { controller, listeners: [listener] };
    const report = (message) => {
      item.listeners = item.listeners.filter((listener) => listener.active());
      for (const listener of item.listeners) status(listener.host, message);
    };
    const loadStarted = performance.now();
    item.promise = (async () => {
      report('Preparing the checked coordinates…');
      const response = await fetch('/api/step/' + task + '/' + step + '/viewer.json', {
        signal: controller.signal,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          body.error || 'Could not load the checked structure (HTTP ' + response.status + ')',
        );
      }
      const reader = response.body.getReader();
      // Content-Length describes compressed bytes; decoded stream bytes are not a percentage of it.
      const total = response.headers.get('Content-Encoding')
        ? 0
        : Number(response.headers.get('Content-Length'));
      const chunks = [];
      let bytes = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        bytes += value.length;
        report(
          'Downloading structure: ' +
            (bytes / 1048576).toFixed(1) +
            ' MiB' +
            (total ? ' / ' + (total / 1048576).toFixed(1) + ' MiB' : ' received'),
        );
      }
      const buffer = new Uint8Array(bytes);
      let offset = 0;
      for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
      }
      report('Parsing coordinates and checking component counts…');
      const decoded = await new Promise((resolve, reject) => {
        const worker = new Worker(
          '/static/viewer-worker.js?v=' + encodeURIComponent(scriptVersion),
        );
        const abort = () => {
          worker.terminate();
          reject(new DOMException('Superseded viewer', 'AbortError'));
        };
        controller.signal.addEventListener('abort', abort, { once: true });
        const finish = () => {
          worker.terminate();
          controller.signal.removeEventListener('abort', abort);
        };
        worker.onmessage = (event) => {
          finish();
          event.data.error ? reject(new Error(event.data.error)) : resolve(event.data);
        };
        worker.onerror = () => {
          finish();
          reject(new Error('Could not decode the structure. Retry or reload this page.'));
        };
        worker.postMessage(buffer.buffer, [buffer.buffer]);
      });
      decoded.loadMs = performance.now() - loadStarted;
      item.bytes = decoded.bytes;
      let retained = [...loads.values()].reduce((sum, value) => sum + (value.bytes || 0), 0);
      for (const [oldKey, old] of loads) {
        if (retained <= MAX_DECODED_BYTES) break;
        if (oldKey === key) continue;
        old.controller.abort(); loads.delete(oldKey); retained -= old.bytes || 0;
      }
      if (decoded.bytes > MAX_DECODED_BYTES) loads.delete(key);
      return decoded;
    })().catch((error) => {
      if (loads.get(key) === item) loads.delete(key);
      throw error;
    });
    loads.set(key, item);
    // Keep three recent artifacts to bound browser memory while allowing step revisits.
    if (loads.size > 3) {
      const oldest = loads.keys().next().value;
      if (oldest !== key) {
        loads.get(oldest).controller.abort();
        loads.delete(oldest);
      }
    }
    return item.promise;
  }
  function invalidate() {
    epoch++;
    for (const item of loads.values()) item.controller.abort();
    loads.clear();
    for (const id of [...scenes.keys()]) releaseScene(id);
    if (window.invalidateFinalReview) window.invalidateFinalReview();
  }
  function box(viewer, matrix, step) {
    // Input/structure boxes are inherited placeholders, not constructed boundaries.
    if (
      !['membrane', 'solvation', 'ions', 'cg_environment', 'cg_solvation', 'cg_system'].includes(step)
    )
      return;
    const points = [];
    const origin =
      step === 'membrane'
        ? [0, 1, 2].map((axis) => -5 * matrix.reduce((sum, row) => sum + row[axis], 0))
        : [0, 0, 0];
    for (let k = 0; k < 8; k++) points.push({ x: origin[0], y: origin[1], z: origin[2] });
    for (let k = 0; k < 8; k++)
      for (let axis = 0; axis < 3; axis++)
        if (k & (1 << axis)) {
          points[k].x += matrix[axis][0] * 10;
          points[k].y += matrix[axis][1] * 10;
          points[k].z += matrix[axis][2] * 10;
        }
    for (let k = 0; k < 8; k++)
      for (let axis = 0; axis < 3; axis++)
        if (!(k & (1 << axis)))
          viewer.addLine({
            start: points[k],
            end: points[k | (1 << axis)],
            color: '#475569',
            linewidth: 2,
          });
  }
  async function render(hostId, step) {
    const host = document.getElementById(hostId);
    if (!host || !state.taskId) return null;
    const task = state.taskId,
      generation = epoch;
    const ticket = Symbol();
    host._viewerTicket = ticket;
    const started = performance.now();
    const active = () =>
      host._viewerTicket === ticket &&
      task === state.taskId &&
      generation === epoch &&
      host.closest('.panel.active');
    let timer = setInterval(() => {
      const el = document.getElementById(hostId + '-loading');
      if (el && active())
        el.dataset.elapsed = Math.floor((performance.now() - started) / 1000) + ' s';
    }, 1000);
    try {
      if (!active()) return null;
      const payload = await load(step, host, active);
      if (!active()) return null;
      status(host, 'Creating protein, lipid, solvent and ion representations…');
      await frame();
      await GMXAssets.viewer();
      if (!active()) return null;
      let scene = scenes.get(hostId);
      if (
        scene &&
        scene.ready &&
        scene.revision === payload.data.revision &&
        scene.step === step &&
        host.querySelector('canvas')
      ) {
        scene.viewer.resize();
        scene.viewer.render();
        await frame();
        await frame();
        if (!active()) return null;
        status(
          host,
          'Structure ready — ' + payload.data.atom_count.toLocaleString() + ' saved atoms.',
        );
        return payload.data;
      }
      let viewer = scene?.viewer;
      if (viewer) viewer.clear();
      else host.replaceChildren();
      host.style.position = 'relative';
      host.style.background = '#fff';
      try {
        if (!viewer)
          viewer = $3Dmol.createViewer(host, { backgroundColor: '#ffffff', antialias: true });
      } catch (_error) {
        throw new Error('3D display unavailable: enable WebGL/hardware acceleration, then retry.');
      }
      const currentScene = { viewer, revision: payload.data.revision, step, models: [] };
      scenes.delete(hostId);
      scenes.set(hostId, currentScene);
      while (scenes.size > 2) releaseScene(scenes.keys().next().value);
      const timing = {loadMs:payload.loadMs, decodeMs:payload.decodeMs, buildMs:0, maxBuildSliceMs:0};
      metrics.set(hostId, timing);
      const canvas = host.querySelector('canvas');
      if (canvas && !canvas._gmxContextWatched)
        canvas.addEventListener(
          'webglcontextlost',
          () => {
            if (scenes.get(hostId)?.viewer !== viewer) return;
            scenes.delete(hostId);
            status(host, 'The graphics context was lost. Retry the viewer.', true);
            if (window.invalidateFinalReview) window.invalidateFinalReview();
          },
          { once: true },
        );
      if (canvas) canvas._gmxContextWatched = true;
      GMXStyle.prepare([]);
      const buildStarted = performance.now();
      let sliceDeadline = performance.now() + 6;
      for (let i = 0; i < payload.groups.length; i++) {
        if (!active()) return null;
        const component = payload.data.components[i], indices = payload.groups[i];
        if (!indices.length) { currentScene.models.push(null); continue; }
        const model = viewer.addModel(); currentScene.models.push(model);
        // 3Dmol addAtoms remaps only bonds inside the supplied batch. Restore
        // the complete component graph after all batches have been appended.
        for (let start = 0; start < indices.length; start += 2048) {
          if (!active()) return null;
          const sliceStarted = performance.now(), chunk = [];
          for (let j=start; j<Math.min(start+2048,indices.length); j++)
            chunk.push(atomAt(payload, indices[j], j, component));
          model.addAtoms(chunk);
          timing.maxBuildSliceMs = Math.max(timing.maxBuildSliceMs, performance.now()-sliceStarted);
          status(host, 'Preparing ' + component.name + ': ' + Math.min(start+2048,indices.length).toLocaleString() +
            ' / ' + indices.length.toLocaleString() + ' displayed atoms…');
          if (performance.now() >= sliceDeadline) {
            await frame();
            sliceDeadline = performance.now() + 6;
          }
        }
        const installed = model.selectedAtoms({});
        for (let start=0; start<indices.length; start+=2048) {
          if (!active()) return null;
          for (let j=start;j<Math.min(start+2048,indices.length);j++) {
            const k=indices[j];
            installed[j].bonds=Array.from(payload.targets.subarray(payload.offsets[k],payload.offsets[k+1]));
            installed[j].bondOrder=installed[j].bonds.map(()=>1);
          }
          if (performance.now() >= sliceDeadline) {
            await frame();
            sliceDeadline = performance.now() + 6;
          }
        }
      }
      timing.buildMs = performance.now()-buildStarted;
      if (!active()) return null;
      for (const bond of payload.crossBonds || [])
        viewer.addLine({ ...bond, color: '#64748b', linewidth: 2 });
      box(viewer, payload.data.box_nm, step);
      if (payload.data.components.some((c) => c.kind === 'MEMBRANE')) viewer.rotate(90, 'x');
      viewer.setSlab(-100000, 100000);
      viewer.resize();
      viewer.zoomTo();
      status(host, 'Drawing the complete system…');
      await frame();
      if (!active()) return null;
      const drawStarted = performance.now();
      viewer.render();
      timing.drawMs = performance.now()-drawStarted;
      timing.totalMs = performance.now()-started;
      await frame();
      await frame();
      if (!active()) return null;
      currentScene.ready = true;
      let controls = document.getElementById(hostId + '-components');
      if (!controls) {
        controls = document.createElement('div');
        controls.id = hostId + '-components';
        controls.className = 'viewer-components';
        host.after(controls);
      }
      controls.replaceChildren();
      payload.data.components.forEach((component, i) => {
        const label = document.createElement('label'),
          toggle = document.createElement('input');
        toggle.type = 'checkbox';
        toggle.checked = true;
        toggle.dataset.viewerControl = 'true';
        toggle.addEventListener('change', () => {
          const model = currentScene.models[i];
          if (model) {
            toggle.checked ? model.show() : model.hide();
            viewer.render();
          }
        });
        label.append(toggle, document.createTextNode(' ' + component.name));
        controls.appendChild(label);
      });
      status(
        host,
        'Structure ready — ' +
          payload.data.atom_count.toLocaleString() +
          ' saved atoms; ' +
          ((performance.now() - started) / 1000).toFixed(1) +
          ' s.',
      );
      return payload.data;
    } catch (error) {
      if (active() && error.name !== 'AbortError') {
        status(host, error.message, true);
        if (hostId !== 'ion-viewer') {
          const retry = document.createElement('button');
          retry.type = 'button';
          retry.textContent = 'Retry viewer';
          retry.addEventListener('click', () => render(hostId, step));
          document.getElementById(hostId + '-loading').appendChild(retry);
        }
      }
      return null;
    } finally {
      clearInterval(timer);
    }
  }
  window.GMXViewer = { render, invalidate, scenes, metrics };
})();

window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('checkpoint_viewer.js');
