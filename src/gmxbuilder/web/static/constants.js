/**
 * GMXBUILDER Frontend Constants
 *
 * Centralized physical constants, default values, color palettes,
 * and residue classification sets.  Import before app.js in index.html.
 *
 * All values in nanometres unless otherwise noted.
 */
(function (window) {
  "use strict";

  // ======================================================================
  // Physical / simulation constants
  // ======================================================================
  var PHYS = {
    /** Default lipid bilayer hydrophobic half-thickness (nm) — POPC */
    DEFAULT_DHH: 3.8,

    /** Water molecule effective volume in a box (nm^3) — TIP3P at 300 K */
    WATER_VOLUME: 0.0299,

    /** Water packing factor (accounts for pre-existing water in built box) */
    WATER_PACKING: 0.96,

    /** Default area per lipid when no registry value is available (nm^2) */
    DEFAULT_APL: 0.65,

    /** Default number of lipids per leaflet when not computed from box */
    DEFAULT_N_PER_LEAFLET: 100,

    /** Membrane box XY padding default (nm, each side) */
    DEFAULT_MEMBRANE_PAD: 2.0,

    /** Box Z padding default for solvation (nm, each side) */
    DEFAULT_BOX_PAD: 1.5,

    /** Protein exclusion margin around membrane in XY plane (nm) */
    PROTEIN_EXCL_MARGIN: 1.2,

    /** Membrane plane grid rendering step (nm) */
    MEMBRANE_PLANE_STEP: 1.5,

    /** Ion exclusion radius from solute atoms (nm) */
    ION_EXCLUSION_RADIUS: 0.35,

    /** Minimum box XY dimension (nm) */
    MIN_BOX_XY: 4.0,

    /** Liquid builder default box size (nm) */
    LIQUID_BOX_DEFAULT: 5.0,

    /** Maximum file upload size (bytes) */
    MAX_UPLOAD_BYTES: 500 * 1024 * 1024,
  };

  // ======================================================================
  // Water model → force-field compatibility
  // ======================================================================
  var WATER_FF = {
    tip3p: "charmm36 / amber / opls-aa",
    spc:   "opls-aa / amber",
    spce:  "opls-aa / amber / charmm36",
    tip4p: "opls-aa / amber",
  };

  // ======================================================================
  // Default ion concentrations (mM) — physiological
  // ======================================================================
  var DEFAULT_ION_CONCS = { NA: 0.15, CL: 0.15 };

  // ======================================================================
  // Default cations / anions when ion script not loaded
  // ======================================================================
  var DEFAULT_CATIONS = ["NA"];
  var DEFAULT_ANIONS  = ["CL"];

  // ======================================================================
  // 3Dmol viewer ion render colours
  // ======================================================================
  var ION_COLORS = {
    NA: "0x3b82f6",  // blue
    K:  "0x8b5cf6",  // purple
    CL: "0xef4444",  // red
    CA: "0x22c55e",  // green
    MG: "0x10b981",  // emerald
    ZN: "0x64748b",  // slate
  };

  // ======================================================================
  // Macaron colour palette — chain colours for 3D viewer
  // ======================================================================
  var MACARON = [
    "0xd4a5c7","0xa5c7d4","0xc7d4a5","0xd4c7a5","0xc7a5d4","0xa5d4c7",
    "0xe8c3c3","0xc3cce8","0xc3e8c3","0xe8e0c3","0xd4b8b8","0xb8c4d4",
    "0xb3d4b3","0xd4d0b3","0xe0c0d0","0xc0d4e0",
  ];

  // ======================================================================
  // Residue classification sets
  // ======================================================================
  var PROTEIN_RESNAMES = new Set([
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET",
    "PHE","PRO","SER","THR","TRP","TYR","VAL","ASH","GLH","CYX","HID","HIE","HIP",
    "LYN","ACE","NME","MSE","SEC","PYL","PTR","SEP","TPO","S1P","T1P","Y1P","ALY","CIR","CSO","CSX","TYS","HSD","HSE","HSP","CYM",
    "MLZ","MLY","M3L","2MR","DA2","SNC","SMC","OCS","KCX","NIY","OAS","SME","HYP","LYZ",
    "NMA","NALA","NGLY","NPRO","NVAI","NLEU","NILE","NASN","NASP","NGLN","NGLU",
    "NMET","NPHE","NTRP","NTYR","NSER","NTHR","NCYS",
  ]);

  var SOLVENT_RESNAMES = new Set([
    "HOH","SOL","WAT","TIP","TIP3","SPC","SPCE",
  ]);

  var ION_RESNAMES = new Set([
    "NA","CL","K","CA","ZN","MG","CD","BR","I","CS","LI",
  ]);

  var LIPID_RESNAMES = new Set([
    "POPC","DPPC","POPE","DOPE","POPG","POPS","DLPC","DMPC","DSPC","SOPC","PIP2",
    "CHOL","CHL1","ERG","CER","DAG","LPS",
    // PIP variants
    "PAPI","SOP2","SOP3",
    // Oxysterols
    "25OHC","27OHC","20AHC","22RHC","24SHC","7KCH",
  ]);

  // ======================================================================
  // Expose
  // ======================================================================
  window.GMX = {
    PHYS: PHYS,
    WATER_FF: WATER_FF,
    DEFAULT_ION_CONCS: DEFAULT_ION_CONCS,
    DEFAULT_CATIONS: DEFAULT_CATIONS,
    DEFAULT_ANIONS: DEFAULT_ANIONS,
    ION_COLORS: ION_COLORS,
    MACARON: MACARON,
    PROTEIN_RESNAMES: PROTEIN_RESNAMES,
    SOLVENT_RESNAMES: SOLVENT_RESNAMES,
    ION_RESNAMES: ION_RESNAMES,
    LIPID_RESNAMES: LIPID_RESNAMES,
  };

})(window);
