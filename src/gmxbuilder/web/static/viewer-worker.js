/* Decode off-thread and transfer compact arrays instead of cloning atom objects. */
self.onmessage = function (event) {
  const started = performance.now();
  try {
    const data = JSON.parse(new TextDecoder().decode(event.data));
    if (data.schema !== 3) throw new Error('Unsupported viewer data version');
    const decode = (value, Type) => {
      const text = atob(value), bytes = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
      return new Type(bytes.buffer);
    };
    const xyz = decode(data.coordinates_A, Float32Array);
    const ids = decode(data.original_indices, Uint32Array);
    const components = decode(data.component_indices, Uint32Array);
    const dictionaries = {};
    for (const key of ['names', 'resnames', 'chains', 'elements', 'resids'])
      dictionaries[key] = {labels:data[key].labels, codes:decode(data[key].codes, Uint32Array)};
    const n = data.display_count;
    if (!Number.isSafeInteger(n) || n < 1 || xyz.length !== n * 3 || ids.length !== n ||
        components.length !== n || Object.values(dictionaries).some(d => d.codes.length !== n))
      throw new Error('Incomplete viewer data');
    const lists = data.components.map(() => []), local = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const group = lists[components[i]];
      if (!group || !Number.isFinite(xyz[i*3] + xyz[i*3+1] + xyz[i*3+2]))
        throw new Error('Invalid component or coordinates');
      for (const dictionary of Object.values(dictionaries))
        if (dictionary.codes[i] >= dictionary.labels.length) throw new Error('Invalid atom dictionary');
      local[i] = group.length; group.push(i);
    }
    const counts = new Uint32Array(n), crossBonds = [];
    for (const [a, b] of data.bonds) {
      if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a >= n || b >= n)
        throw new Error('Invalid display bond');
      if (components[a] === components[b]) { counts[a]++; counts[b]++; }
      else crossBonds.push({start:{x:xyz[a*3],y:xyz[a*3+1],z:xyz[a*3+2]},
                            end:{x:xyz[b*3],y:xyz[b*3+1],z:xyz[b*3+2]}});
    }
    const offsets = new Uint32Array(n+1);
    for (let i = 0; i < n; i++) offsets[i+1] = offsets[i] + counts[i];
    const targets = new Uint32Array(offsets[n]), cursor = offsets.slice();
    for (const [a, b] of data.bonds) if (components[a] === components[b]) {
      targets[cursor[a]++] = local[b]; targets[cursor[b]++] = local[a];
    }
    const groups = lists.map((list,i) => {
      if (list.length !== data.components[i].display_atoms) throw new Error('Component count mismatch');
      return Uint32Array.from(list);
    });
    const arrays = [xyz,ids,components,offsets,targets,...groups,...Object.values(dictionaries).map(d=>d.codes)];
    self.postMessage({data:{revision:data.revision,source_step:data.source_step,
      resolution:data.resolution,bond_sources:data.bond_sources,atom_count:data.atom_count,
      display_count:n,box_nm:data.box_nm,components:data.components},
      groups, xyz, ids, dictionaries, offsets, targets, crossBonds,
      bytes:arrays.reduce((sum,a)=>sum+a.byteLength,0), decodeMs:performance.now()-started
    }, arrays.map(a=>a.buffer));
  } catch (error) { self.postMessage({error:error.message || String(error)}); }
};
