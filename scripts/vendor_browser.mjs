// Reproduce the exact browser dependencies installed by npm ci.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const copy=async(from,to)=>{await fs.mkdir(path.dirname(to),{recursive:true});await fs.copyFile(from,to);};
await copy(path.join(root,'node_modules/chess.js/dist/esm/chess.js'),path.join(root,'web/vendor/chess.js'));
await copy(path.join(root,'node_modules/chess.js/LICENSE'),path.join(root,'web/vendor/chess-LICENSE'));
for(const name of ['ort.wasm.min.js','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']) {
  await copy(path.join(root,'node_modules/onnxruntime-web/dist',name),path.join(root,'web/vendor/ort',name));
}
console.log('Vendored chess.js 1.4.0 and ONNX Runtime Web 1.31.1; retain web/vendor/ort/LICENSE.');
