/* Frozen Maia policy, running entirely in a browser worker. */
'use strict';
let runtime, core, config, table, session, ready;
let queue = Promise.resolve();
const post = (type,data={}) => self.postMessage({type,...data});
const hash = async data => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');

async function verifiedFetch(url, expected, progress) {
  let cache;
  try {
    cache = await caches.open('justin-chess-fp32-v1');
    const hit = await cache.match(url);
    if (hit) {
      const data = new Uint8Array(await hit.arrayBuffer());
      if (data.byteLength === expected.bytes && await hash(data) === expected.sha256) { progress?.(data.byteLength); return data; }
      await cache.delete(url);
    }
  } catch { cache = null; }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed (${response.status}). Please try again.`);
  const reader = response.body.getReader();
  const chunks = []; let size = 0;
  for (;;) {
    const {done,value} = await reader.read();
    if (done) break;
    chunks.push(value); size += value.byteLength; progress?.(value.byteLength);
  }
  const data = new Uint8Array(size); let offset=0;
  for (const chunk of chunks) { data.set(chunk,offset); offset+=chunk.byteLength; }
  if (data.byteLength !== expected.bytes || await hash(data) !== expected.sha256) throw new Error('The download was incomplete. Please try again.');
  if (cache) { try { await cache.put(url,new Response(data,{headers:{'Content-Type':'application/octet-stream'}})); } catch {} }
  return data;
}

async function init(base) {
  core = await import(new URL('core.js',base).href);
  importScripts(new URL('vendor/ort/ort.wasm.min.js',base).href);
  runtime = ort;
  runtime.env.wasm.wasmPaths = new URL('vendor/ort/',base).href;
  runtime.env.wasm.numThreads = 1;
  const response = await fetch(new URL('manifest.json',base));
  if (!response.ok) throw new Error('Could not load the bot. Please try again.');
  config = await response.json();
  let loaded = 0;
  const total = config.model.bytes + config.table.bytes;
  const progress = bytes => { loaded+=bytes; post('progress',{loaded,total}); };
  const model = new Uint8Array(config.model.bytes); let offset = 0;
  for (const part of config.model.parts) {
    const data = await verifiedFetch(new URL(part.file,base).href,part,progress);
    model.set(data,offset); offset+=data.byteLength;
  }
  if (await hash(model) !== config.model.sha256) throw new Error('The model download could not be verified.');
  const tableData = await verifiedFetch(new URL(config.table.file,base).href,config.table,progress);
  table = JSON.parse(new TextDecoder().decode(tableData));
  post('compiling');
  session = await runtime.InferenceSession.create(model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  // Include compilation and the first inference in loading, rather than the first move.
  await infer(new Float32Array(64*config.channels));
  post('ready',{modelBytes:config.model.bytes});
}

async function infer(tokens) {
  const outputs = await session.run({tokens:new runtime.Tensor('float32',tokens,[1,64,config.channels])});
  return outputs.logits.data;
}

async function process(message) {
  if (message.type === 'init') { ready = init(message.base); await ready; return; }
  if (!ready) throw new Error('The bot is not loaded yet.');
  await ready;
  const {game,history} = core.restoreGame(message.initialFen,message.moves || []);
  const start = performance.now();
  if (message.type === 'evaluate') {
    const logits = await infer(core.tokensFromHistory(history,config));
    const legalProbabilities = core.legalProbabilities(game,logits);
    const pool = await core.policyPool(game,history,table,config,async()=>logits);
    post('evaluation',{id:message.id,legalProbabilities,...pool,ms:performance.now()-start});
  } else if (message.type === 'move') {
    if (core.finished(game)) throw new Error('The game has ended. Start a new game to play again.');
    const pool = await core.policyPool(game,history,table,config,infer);
    post('move',{id:message.id,move:core.sample(pool.probabilities),ms:performance.now()-start,source:pool.source});
  }
}

self.onmessage = event => {
  queue = queue.then(()=>process(event.data)).catch(error => post('error',{id:event.data.id,message:error.message || String(error)}));
};
