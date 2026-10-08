"""Browser-independent timing contracts with actual application JavaScript."""

import shutil
import subprocess
from pathlib import Path

import pytest

HTTP = Path(__file__).parents[1] / "src/gmxbuilder/web/static/http.js"


def test_read_timeout_write_no_replay_and_single_inflight_poll():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to run the actual JavaScript lifecycle")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
global.window=global;global.document={hidden:false};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));
(async()=>{
 let calls=0;
 global.fetch=(url,options)=>{calls++;return new Promise((resolve,reject)=>{
   options.signal.addEventListener('abort',()=>reject(new DOMException('abort','AbortError')));
 });};
 await assert.rejects(GMXHttp.json('/api/options',{}, {timeoutMs:15}),/timed out/);
 assert.equal(calls,1);
 calls=0;global.fetch=async()=>{calls++;return new Response('{}',{status:503});};
 await assert.rejects(GMXHttp.json('/api/build',{method:'POST'}));assert.equal(calls,1);
 let release, started=0, active=0, peak=0;
 const gate=new Promise(r=>release=r);
 const poll=GMXPoll.start(async()=>{
   started++;peak=Math.max(peak,++active);await gate;active--;
 },5,true);
 await new Promise(r=>setTimeout(r,35));assert.equal(started,1);release();
 await new Promise(r=>setTimeout(r,20));GMXPoll.stop(poll);
 assert.equal(peak,1);const stopped=started;await new Promise(r=>setTimeout(r,20));
 assert.equal(started,stopped);
 calls=0;global.fetch=async()=>{
   calls++;await new Promise(r=>setTimeout(r,10));return new Response('{"ready":true}');
 };
 const results=await Promise.all([GMXPoll.read('/status'),GMXPoll.read('/status')]);
 assert.equal(calls,1);assert.equal(results[0].ready,true);
})();
"""
    result = subprocess.run(
        [node, "-e", script, str(HTTP)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
