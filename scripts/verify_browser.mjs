import {chromium} from 'playwright';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const types={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.css':'text/css','.wasm':'application/wasm','.bin':'application/octet-stream'};
let chunkRequests=0;
const server=http.createServer(async(req,res)=>{
 try{
  const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const target=path.resolve(root,'.'+pathname);
  if(!target.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  const stat=await fs.stat(target);const file=stat.isDirectory()?path.join(target,'index.html'):target;
  const data=await fs.readFile(file);
  if(file.endsWith('.bin'))chunkRequests++;
  res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Content-Length':data.length});res.end(data);
 }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const address='http://127.0.0.1:'+server.address().port+'/web/';
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{})});
try {
const page=await browser.newPage({viewport:{width:1000,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(address);
const result=await page.evaluate(async()=>{
 const core=await import('./core.js');
 const config=await (await fetch('./manifest.json')).json();
 const table=await (await fetch('./table.json')).json();
 const fixtures=await (await fetch('../tests/browser-fixtures.json')).json();
 for(const key of Object.keys(table)){
  const actual=core.positionKey(new core.Chess(key+' 0 1'));
  if(actual!==key)throw new Error('Table position key mismatch: '+key+' / '+actual);
 }
 for(const fixture of fixtures){
  const {game,history}=core.restoreGame(fixture.initialFen,fixture.moves);
  const actual=Array.from(core.tokensFromHistory(history,config)).flatMap((value,i)=>value?[i]:[]);
  if(JSON.stringify(actual)!==JSON.stringify(fixture.tokenOnes))throw new Error('Token mismatch at '+fixture.fen);
 }
 const worker=new Worker('worker.js');let next=0;const pending=new Map();
 const ready=new Promise((resolve,reject)=>{
  worker.onerror=e=>reject(new Error(e.message));
  worker.onmessage=e=>{const m=e.data;if(m.type==='ready')resolve();else if(m.type==='evaluation'){pending.get(m.id)?.resolve(m);pending.delete(m.id);}else if(m.type==='error'){reject(new Error(m.message));pending.get(m.id)?.reject(new Error(m.message));}};
 });
 const start=performance.now();worker.postMessage({type:'init',base:new URL('./',location.href).href});await ready;
 const loadMs=performance.now()-start;let maxModelError=0,maxPolicyError=0;const times=[];
 for(const fixture of fixtures){
  const id=++next;const task=new Promise((resolve,reject)=>pending.set(id,{resolve,reject}));
  worker.postMessage({type:'evaluate',id,initialFen:fixture.initialFen,moves:fixture.moves});
  const actual=await task;times.push(actual.ms);
  for(const [move,p] of Object.entries(fixture.legalProbabilities))maxModelError=Math.max(maxModelError,Math.abs(p-actual.legalProbabilities[move]));
  if(Object.keys(fixture.playableProbabilities).sort().join()!==Object.keys(actual.probabilities).sort().join())throw new Error('Playable move set differs: '+fixture.fen);
  for(const [move,p] of Object.entries(fixture.playableProbabilities))maxPolicyError=Math.max(maxPolicyError,Math.abs(p-actual.probabilities[move]));
  if(actual.source!==fixture.source)throw new Error('Policy source differs');
 }
 worker.terminate();times.sort((a,b)=>a-b);
 if(maxModelError>1e-5||maxPolicyError>1e-5)throw new Error('Browser probability parity exceeded tolerance');
 return {positions:fixtures.length,verifiedTableKeys:Object.keys(table).length,tokenization:'exact match',maxModelProbabilityError:maxModelError,maxPolicyProbabilityError:maxPolicyError,loadLocalMs:loadMs,medianEvaluationMs:times[Math.floor(times.length/2)]};
});
console.log(JSON.stringify(result,null,2));
const coldChunkRequests=chunkRequests;
await page.getByRole('button',{name:'Play Justin',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('[data-status]').textContent==='Your turn.',{timeout:120000});
await page.locator('[data-square="e2"]').click();await page.locator('[data-square="e4"]').click();
await page.waitForFunction(()=>document.querySelector('[data-moves]').textContent.trim().split(/\s+/).length>=3 && document.querySelector('[data-status]').textContent==='Your turn.',{timeout:30000});
await page.getByRole('button',{name:'Play as Black',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('[data-status]').textContent==='Your turn.',{timeout:30000});
if(await page.locator('[data-board] button').first().getAttribute('data-square')!=='h1')throw new Error('Black orientation failed');
await page.getByRole('button',{name:'New game',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('[data-status]').textContent==='Your turn.',{timeout:30000});
if(process.env.BROWSER_SCREENSHOT_DIR){await fs.mkdir(process.env.BROWSER_SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.BROWSER_SCREENSHOT_DIR,'desktop.png'),fullPage:true});}
await page.setViewportSize({width:375,height:850});
if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw new Error('Mobile horizontal overflow');
if(process.env.BROWSER_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.BROWSER_SCREENSHOT_DIR,'mobile.png'),fullPage:true});
if(chunkRequests!==coldChunkRequests)throw new Error('Model chunks were downloaded again instead of using browser cache');
console.log('Cache: repeated model initialization reused verified cached chunks.');
if(errors.length)throw new Error(errors.join('\n'));
console.log('UI: loading, White move and model reply, Black orientation, new game, mobile width passed.');
} finally { await browser.close(); await new Promise(resolve=>server.close(resolve)); }
