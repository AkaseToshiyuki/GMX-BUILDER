/* Load optional libraries once, at their first use. Failed loads can be retried. */
(function () {
  const pending = new Map();
  function load(name, url, present) {
    if (present()) return Promise.resolve();
    if (pending.has(name)) return pending.get(name);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      let timer;
      const failed = () => {
        clearTimeout(timer); script.remove(); pending.delete(name);
        reject(new Error('The ' + name + ' library could not load. Retry this preview.'));
      };
      script.src = url;
      script.onload = () => { clearTimeout(timer); present() ? resolve() : failed(); };
      script.onerror = failed;
      timer = setTimeout(failed, 20000);
      document.head.appendChild(script);
    });
    pending.set(name, promise);
    return promise;
  }
  window.GMXAssets = {
    viewer: () => load('3D viewer', '/static/vendor/3dmol-2.5.5/3Dmol-min.js', () => !!window.$3Dmol),
    smiles: () => load('SMILES viewer', '/static/vendor/smiles-drawer-2.0.3/smiles-drawer.min.js', () => !!window.SmilesDrawer)
  };
})();
