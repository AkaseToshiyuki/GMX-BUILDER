/* One representation and identity-based palette for every molecular viewer. */
(function () {
  'use strict';
  const moleculePalette = [
    '0xf59e0b', '0xef4444', '0x10b981', '0x8b5cf6', '0x06b6d4',
    '0xf97316', '0xec4899', '0x6366f1', '0x14b8a6', '0x84cc16',
    '0xeab308', '0xd946ef', '0x0ea5e9', '0x78716c', '0x65a30d',
  ];
  const lipidPalette = ['0xd97706', '0xa855f7', '0x16a34a', '0xdb2777', '0x0891b2'];
  const nucleic = new Set(['A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'RA', 'RC', 'RG', 'RU']);
  let task = null;
  let chains = new Map(), molecules = new Map(), lipids = new Map();
  let polymerResidues = new Map();
  let knownLipids = new Set(GMX.LIPID_RESNAMES);
  function color(map, key, palette) {
    key = String(key || '');
    if (!map.has(key)) map.set(key, palette[map.size % palette.length]);
    return map.get(key);
  }
  function prepare(atoms) {
    const current = window.state?.taskId || '';
    if (current !== task) {
      task = current;
      chains = new Map(); molecules = new Map(); lipids = new Map();
    }
    const info = window.state?.pdbInfo || {};
    polymerResidues = new Map();
    const registry = typeof _lipidPickerData === 'undefined' ? [] : _lipidPickerData.lipids;
    knownLipids = new Set([...GMX.LIPID_RESNAMES, ...registry.map(lipid => lipid.name)]);
    // Seed from input identity, so opening a later step first gives the same colors.
    for (const row of info.fragment_choices || info.checked_sequences || info.sequences || []) {
      color(chains, row.chain_id, GMX.MACARON);
      for (const residue of row.residues || [])
        polymerResidues.set(JSON.stringify([row.chain_id, residue.resname]),
          residue.is_nucleic ? 'NUCLEIC_ACID' : 'PROTEIN');
    }
    for (const row of (info.selection_info || info).small_molecules || [])
      color(molecules, row.resname, moleculePalette);
    for (const atom of atoms)
      if (kindOf(atom) === 'PROTEIN' || kindOf(atom) === 'NUCLEIC_ACID')
        color(chains, atom.chain, GMX.MACARON);
  }
  function kindOf(atom) {
    const name = atom.resn;
    const polymer = polymerResidues.get(JSON.stringify([atom.chain, name]));
    if (polymer) return polymer;
    if (GMX.PROTEIN_RESNAMES.has(name)) return 'PROTEIN';
    if (nucleic.has(name) || /^(?:D[ACGTU]|R?[ACGU])[35]$/.test(name)) return 'NUCLEIC_ACID';
    if (GMX.SOLVENT_RESNAMES.has(name) || name === 'W') return 'SOLVENT';
    if (GMX.ION_RESNAMES.has(name)) return 'IONS';
    if (knownLipids.has(name)) return 'MEMBRANE';
    return 'LIGAND';
  }
  function representation(kind, identity, coarse) {
    if (kind === 'PROTEIN' || kind === 'NUCLEIC_ACID') {
      const shade = color(chains, identity, GMX.MACARON);
      return coarse
        ? { sphere: { radius: 0.25, color: shade }, stick: { radius: 0.1, color: shade } }
        : { cartoon: { color: shade, style: 'trace', thickness: 0.28 } };
    }
    if (kind === 'SOLVENT')
      // Coordinates/radii are in Angstroms. Subpixel translucent spheres vanish
      // at whole-box zoom; keep water visible while leaving the solute readable.
      return { sphere: { radius: coarse ? 0.35 : 0.22, color: '0x3b82f6', opacity: 0.65 } };
    if (kind === 'IONS')
      return { sphere: { radius: 0.3, color: GMX.ION_COLORS[identity] || '0x94a3b8' } };
    if (kind === 'MEMBRANE') {
      const shade = color(lipids, identity, lipidPalette);
      const style = { stick: { radius: 0.12, color: shade } };
      if (coarse) style.sphere = { radius: 0.22, color: shade };
      return style;
    }
    const shade = color(molecules, identity, moleculePalette);
    return { stick: { radius: 0.18, color: shade },
      sphere: { radius: 0.25, color: shade, opacity: 0.8 } };
  }
  function apply(target, atoms, options = {}) {
    prepare(atoms);
    target.setStyle({}, {});
    const groups = new Map();
    for (const atom of atoms) {
      const kind = options.kind || kindOf(atom);
      const polymer = kind === 'PROTEIN' || kind === 'NUCLEIC_ACID';
      const identity = polymer ? atom.chain : atom.resn;
      const key = JSON.stringify([kind, identity]);
      if (!groups.has(key)) groups.set(key, { kind, identity, polymer, residues: new Set() });
      groups.get(key).residues.add(atom.resn);
    }
    for (const group of groups.values()) {
      const { kind, identity, polymer, residues } = group;
      const selection = { resn: [...residues] };
      if (polymer) selection.chain = identity;
      if (kind === 'SOLVENT' && !options.coarse) selection.elem = 'O';
      if (polymer && options.onlyChains && !options.onlyChains.has(identity)) continue;
      if (!polymer && options.moleculeState?.[identity]?.included === false) continue;
      target.setStyle(selection, representation(kind, identity, options.coarse));
    }
    window._pdbChainColors = Object.fromEntries(chains);
  }
  function pdbAtoms(viewer, pdb) {
    const atoms = viewer.selectedAtoms ? viewer.selectedAtoms({}) : [];
    if (atoms.length) return atoms;
    // Also supports lightweight previews that have no selectedAtoms accessor.
    return String(pdb || '').split('\n').filter(line => /^(ATOM  |HETATM)/.test(line))
      .map(line => ({ chain: line.slice(21, 22).trim(), resn: line.slice(17, 21).trim(),
        atom: line.slice(12, 16).trim(), elem: line.slice(76, 78).trim() }));
  }
  function atomStyle(atom, kind, coarse) {
    kind = kind || kindOf(atom);
    const polymer = kind === 'PROTEIN' || kind === 'NUCLEIC_ACID';
    return representation(kind, polymer ? atom.chain : atom.resn, coarse);
  }
  window.GMXStyle = { apply, prepare, representation, pdbAtoms, atomStyle };
})();

window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('viewer_style.js');
