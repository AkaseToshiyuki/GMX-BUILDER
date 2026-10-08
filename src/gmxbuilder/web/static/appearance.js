/* Presentation only. No task state, API calls, controls or workflow changes.
 * ?appearance=classic restores the retained classic stylesheet in this tab;
 * ?appearance=dark returns to the dark workspace. The choice survives the
 * existing wizard's history updates without changing its routing code.
 */
(function () {
  'use strict';
  var sheet = document.getElementById('dark-appearance');
  var appearance = sheet ? sheet.dataset.default : 'classic';
  var requested = new URLSearchParams(window.location.search).get('appearance');
  var valid = function (value) { return value === 'classic' || value === 'dark'; };
  try {
    if (valid(requested)) sessionStorage.setItem('gmxbuilder-appearance', requested);
    var saved = sessionStorage.getItem('gmxbuilder-appearance');
    if (valid(saved)) appearance = saved;
  } catch (_error) { /* Storage may be disabled; the explicit URL still works. */ }
  if (valid(requested)) appearance = requested;
  if (sheet) sheet.media = appearance === 'dark' ? 'all' : 'not all';
  document.documentElement.dataset.appearance = appearance;

  // Only the scene background changes. Atom colors, geometry, selection,
  // camera, clipping and every scientific rendering option stay with callers.
  window.gmxViewerBackground = function () {
    return '#ffffff';
  };
}());

window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('appearance.js');
