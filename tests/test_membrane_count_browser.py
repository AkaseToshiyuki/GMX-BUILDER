"""Checked leaflet counts must replace starting estimates, including equal mixtures."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_actual_leaflet_counts_are_visible_and_invalidated_on_edits(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script("return typeof updateLipidCounts === 'function'")
    )
    page.execute_async_script("selectTaskType('pure-membrane').then(arguments[0])")
    page.execute_async_script("""
      const done=arguments[0], originalFetch=window.fetch;
      document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-membrane'));
      document.getElementById('n-lipids-per-leaflet').value=100;
      document.getElementById('ff-protein').value='charmm36m';
      document.getElementById('ff-lipid').innerHTML='<option value="charmm36m">CHARMM36m</option>';
      _mixUpper=[{name:'POPC',ratio:100}]; _asymmetric=false;
      window.fetch=async function(url,options) {
        if(String(url)==='/api/lipid-library-list')
          return {ok:true,json:async()=>({library_version:4,entries:[
            {lipid_name:'POPC',lipid_ff:'charmm36m',ready:true}
          ]})};
        if(String(url).endsWith('/membrane') && options && options.method==='POST')
          return {ok:true,json:async()=>({status:'ok',metrics:{
            box_dimensions_nm:[10,10,8],membrane:{
              n_lipids_upper:117,n_lipids_lower:100,requested_lipids_per_leaflet:100,
              lipid_counts_upper:{POPC:117},lipid_counts_lower:{POPC:100}}}})};
        return originalFetch(url,options);
      };
      checkComposition().finally(()=>{window.fetch=originalFetch;done();});
    """)
    result = page.execute_script("""
      return {
        summary:document.getElementById('membrane-actual-counts').textContent,
        upper:document.getElementById('upper-lipid-counts').textContent,
        lower:document.getElementById('lower-lipid-counts').textContent,
        lowerVisible:document.getElementById('lower-lipid-counts').getClientRects().length>0,
        siblings:document.getElementById('lower-lipid-counts').parentElement ===
          document.getElementById('upper-lipid-counts').parentElement,
        warning:document.getElementById('membrane-count-warning').textContent,
        progress:document.getElementById('step-progress-check-composition-btn').dataset.state
      };
    """)
    assert "upper 117, lower 100" in result["summary"]
    assert "Actual Upper lipids: 117" in result["upper"]
    assert "Actual Lower lipids: 100" in result["lower"]
    assert result["lowerVisible"]
    assert result["siblings"]
    assert result["progress"] == "warning"
    assert "Requested 100" in result["warning"]
    assert "actual lipid ratios in both leaflets" in result["warning"]
    # Matching counts remain an ordinary successful Check.
    matching = page.execute_script("""
      _membraneActualCounts.n_lipids_upper=100;
      updateCompositionStatus();
      return {
        state:document.getElementById('step-progress-check-composition-btn').dataset.state,
        hidden:document.getElementById('membrane-count-warning').classList.contains('hidden')
      };
    """)
    assert matching == {"state": "done", "hidden": True}
    cleared = page.execute_script("""
      document.getElementById('n-lipids-per-leaflet').value=120;
      document.getElementById('n-lipids-per-leaflet')
        .dispatchEvent(new Event('input', {bubbles:true}));
      return document.getElementById('membrane-actual-counts').textContent;
    """)
    assert "Initial count estimate" in cleared
    assert "117" not in cleared


def test_molecular_viewers_are_two_to_one_at_desktop_and_mobile_widths(page):
    original = page.get_window_size()
    try:
        for width in (1440, 600):
            page.set_window_size(width, 1000)
            bounds = page.execute_script("""
              return [...document.querySelectorAll('.viewer-box,.viewer-container,.mol-viewer')]
                .map(host=>{
                  const panel=host.closest('.panel');
                  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p===panel));
                  for(let node=host.parentElement;node;node=node.parentElement)
                    node.classList.remove('hidden');
                  const rect=host.getBoundingClientRect();
                  return {id:host.id,width:rect.width,height:rect.height};
                });
            """)
            assert len(bounds) == 10
            for box in bounds:
                assert box["width"] > 100, box
                assert abs(box["width"] / 2 - box["height"]) < 1, box
    finally:
        page.set_window_size(original["width"], original["height"])
