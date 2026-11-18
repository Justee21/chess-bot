import { Chess, finished, moveObject, toUci } from './core.js';

const host = document.querySelector('[data-justin-chess]');
const $ = selector => host.querySelector(selector);
const boardElement = $('[data-board]');
const status = $('[data-status]');
const symbols = {k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const names = {k:'king',q:'queen',r:'rook',b:'bishop',n:'knight',p:'pawn'};
let game = new Chess(), human='w', selected=null, lastMove=null;
let worker, loaded=false, loading=false, thinking=false, serial=0, activeRequest=0, pendingPromotion;

function turnMessage() {
  if (game.isCheckmate()) return game.turn() === human ? 'Checkmate. Justin wins.' : 'Checkmate. You win.';
  if (finished(game)) return 'Draw. Play again?';
  return `${game.turn() === human ? 'Your turn' : 'Justin is thinking…'}${game.isCheck() ? ' — check' : ''}${game.turn() === human ? '.' : ''}`;
}

function render() {
  const focused = document.activeElement?.dataset?.square;
  const legal = selected ? game.moves({square:selected,verbose:true}) : [];
  const destinations = new Set(legal.map(move=>move.to));
  const fragment = document.createDocumentFragment();
  for (let row=0;row<8;row++) for(let col=0;col<8;col++) {
    const rank = human === 'w' ? 7-row : row;
    const file = human === 'w' ? col : 7-col;
    const square = String.fromCharCode(97+file)+(rank+1);
    const piece = game.get(square);
    const cell = document.createElement('button');
    cell.type='button'; cell.dataset.square=square;
    cell.className='square'+((rank+file)%2===0?' dark':'')+(selected===square?' selected':'')+(destinations.has(square)?' legal':'')+(piece?' has-piece':'')+((lastMove?.from===square||lastMove?.to===square)?' last':'');
    cell.setAttribute('aria-label',`${square}${piece ? ` ${piece.color==='w'?'white':'black'} ${names[piece.type]}` : ' empty'}`);
    cell.setAttribute('aria-pressed',String(selected===square));
    if (piece) {
      const glyph=document.createElement('span');glyph.className='piece'+(piece.color==='w'?' white':'');glyph.textContent=symbols[piece.type];glyph.setAttribute('aria-hidden','true');cell.append(glyph);
    }
    if(row===7){const label=document.createElement('span');label.className='coord';label.textContent=String.fromCharCode(97+file);cell.append(label);}
    if(col===0){const label=document.createElement('span');label.className='coord rank';label.textContent=String(rank+1);cell.append(label);}
    cell.addEventListener('click',()=>clickSquare(square));
    fragment.append(cell);
  }
  boardElement.replaceChildren(fragment);
  if(focused) boardElement.querySelector(`[data-square="${focused}"]`)?.focus({preventScroll:true});
  const history=game.history();
  $('[data-moves]').replaceChildren(...Array.from({length:Math.ceil(history.length/2)},(_,i)=>{
    const span=document.createElement('span');span.className='move-pair';span.textContent=`${i+1}. ${history[2*i]} ${history[2*i+1]||''}`;return span;
  }));
  $('[data-moves]').scrollTop=$('[data-moves]').scrollHeight;
  if(loaded) status.textContent=turnMessage();
}

function requestMove() {
  if(!loaded||thinking||game.turn()===human||finished(game)) return;
  thinking=true;activeRequest=++serial;
  status.textContent=turnMessage();
  worker.postMessage({type:'move',id:activeRequest,initialFen:new Chess().fen(),moves:game.history({verbose:true}).map(toUci)});
}

function commit(move) {
  lastMove=game.move(move);selected=null;render();requestMove();
}

function clickSquare(square) {
  if(!loaded||thinking||game.turn()!==human||finished(game)||pendingPromotion) return;
  const piece=game.get(square);
  if(selected) {
    const candidates=game.moves({square:selected,verbose:true}).filter(move=>move.to===square);
    if(candidates.length) {
      if(candidates.some(move=>move.promotion)) { choosePromotion(candidates);return; }
      commit({from:selected,to:square});return;
    }
  }
  selected=piece?.color===human && selected!==square ? square : null;
  render();
}

function choosePromotion(candidates) {
  pendingPromotion=true;
  const dialog=$('[data-promotion]');dialog.hidden=false;
  const choices=$('[data-promotion-options]');choices.replaceChildren();
  for(const promotion of ['q','r','b','n']) {
    const candidate=candidates.find(move=>move.promotion===promotion);if(!candidate)continue;
    const button=document.createElement('button');button.type='button';button.textContent=symbols[promotion];button.setAttribute('aria-label',names[promotion]);
    button.addEventListener('click',()=>{pendingPromotion=false;dialog.hidden=true;commit({from:candidate.from,to:candidate.to,promotion});});choices.append(button);
  }
  choices.querySelector('button')?.focus();
}

function reset(color=human) {
  human=color;activeRequest=++serial;thinking=false;
  game=new Chess();selected=null;lastMove=null;pendingPromotion=false;$('[data-promotion]').hidden=true;
  for(const button of host.querySelectorAll('[data-color]')) button.setAttribute('aria-pressed',String(button.dataset.color===human));
  render();requestMove();
}

function fail(message) {
  loading=false;thinking=false;loaded=false;
  status.textContent=`${message} You can try loading the bot again.`;
  $('[data-download]').hidden=true;
  $('[data-start]').hidden=false;$('[data-start]').textContent='Try again';
  if(worker){worker.terminate();worker=null;}
}

function start() {
  $('[data-game]').hidden=false;$('[data-start]').hidden=true;render();
  if(loaded){requestMove();return;}if(loading)return;
  if(typeof WebAssembly!=='object'||typeof Worker!=='function'||!crypto.subtle){fail('This browser cannot run the bot. Try a current browser.');return;}
  loading=true;status.textContent='Downloading Justin for your first game…';
  worker=new Worker(new URL('worker.js',import.meta.url));
  worker.onerror=event=>fail(event.message || 'The bot could not start.');
  worker.onmessage=event=>{
    const message=event.data;
    if(message.type==='progress') {
      $('[data-download]').hidden=false;
      $('[data-download] progress').value=100*message.loaded/message.total;
      status.textContent=`Loading the bot: ${(message.loaded/1e6).toFixed(0)} / ${(message.total/1e6).toFixed(0)} MB`;
    } else if(message.type==='compiling') {status.textContent='Getting Justin ready…';$('[data-download]').hidden=true;}
    else if(message.type==='ready') {loaded=true;loading=false;$('[data-download]').hidden=true;render();requestMove();}
    else if(message.type==='move') {
      if(message.id!==activeRequest)return;
      thinking=false;
      try{lastMove=game.move(moveObject(message.move));render();}catch{fail('The bot could not finish that move.');}
    } else if(message.type==='error') {
      if(message.id && message.id!==activeRequest)return;
      fail(message.message);
    }
  };
  worker.postMessage({type:'init',base:new URL('.',import.meta.url).href});
}

$('[data-start]').addEventListener('click',start);
$('[data-new]').addEventListener('click',()=>reset());
for(const button of host.querySelectorAll('[data-color]')) button.addEventListener('click',()=>reset(button.dataset.color));
boardElement.addEventListener('keydown',event=>{
  if(event.key==='Escape'){selected=null;render();return;}
  const direction={ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8}[event.key];
  if(direction===undefined)return;
  event.preventDefault();const cells=[...boardElement.children];
  const index=cells.indexOf(document.activeElement);cells[Math.max(0,Math.min(63,index+direction))]?.focus();
});
const parameters=new URLSearchParams(location.search);
if(parameters.has('embed')) {
  document.body.classList.add('embedded');
  const reportSize=()=>parent.postMessage({type:'justin-chess-size',height:Math.ceil(host.getBoundingClientRect().height+4)},location.origin);
  new ResizeObserver(reportSize).observe(host);
  reportSize();
}
if(parameters.has('autostart'))start();
