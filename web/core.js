import { Chess } from './vendor/chess.js';
export { Chess };

const pieces = 'pnbrqk';
export const moveObject = uci => ({from: uci.slice(0,2), to: uci.slice(2,4), ...(uci[4] ? {promotion: uci[4]} : {})});
export const toUci = move => move.from + move.to + (move.promotion || '');
export const positionKey = game => game.fen().split(' ').slice(0,4).join(' ');

export function tokensFromHistory(history, config) {
  const recent = history.slice(-config.history);
  while (recent.length < config.history) recent.unshift(recent[0]);
  const data = new Float32Array(64 * config.channels);
  recent.forEach((fen, frame) => {
    const [placement, turn] = fen.split(' ');
    let rank = 7, file = 0;
    for (const c of placement) {
      if (c === '/') { rank--; file = 0; continue; }
      if (c >= '1' && c <= '8') { file += Number(c); continue; }
      const black = turn === 'b';
      const square = (black ? 7-rank : rank) * 8 + file;
      const originallyWhite = c === c.toUpperCase();
      const orientedWhite = black ? !originallyWhite : originallyWhite;
      const channel = pieces.indexOf(c.toLowerCase()) + (orientedWhite ? 0 : 6);
      data[square * config.channels + frame * 12 + channel] = 1;
      file++;
    }
  });
  return data;
}

export function moveIndex(uci, turn) {
  const square = text => (text.charCodeAt(0)-97) + 8 * (turn === 'w' ? Number(text[1])-1 : 8-Number(text[1]));
  if (uci.length === 5) return 4096 + ((uci.charCodeAt(0)-97)*8 + uci.charCodeAt(2)-97)*4 + 'qrbn'.indexOf(uci[4]);
  return square(uci.slice(0,2))*64 + square(uci.slice(2,4));
}

export function legalProbabilities(game, logits) {
  const legal = game.moves({verbose:true}).map(toUci).sort((a,b) => moveIndex(a,game.turn())-moveIndex(b,game.turn()));
  const max = Math.max(...legal.map(move => logits[moveIndex(move,game.turn())]));
  const weights = legal.map(move => Math.exp(logits[moveIndex(move,game.turn())]-max));
  const total = weights.reduce((a,b) => a+b,0);
  return Object.fromEntries(legal.map((move,i) => [move, weights[i]/total]));
}

export function finished(game) {
  return game.isGameOver() || Number(game.fen().split(' ')[4]) >= 100;
}

function playable(game, distribution) {
  const result = {};
  for (const [uci,p] of Object.entries(distribution)) {
    if (!(p > 0)) continue;
    game.move(moveObject(uci));
    const draws = finished(game) && !game.isCheckmate();
    game.undo();
    if (!draws) result[uci] = p;
  }
  return result;
}

function normalize(distribution) {
  const total = Object.values(distribution).reduce((a,b) => a+b,0);
  return Object.fromEntries(Object.entries(distribution).map(([move,p]) => [move,p/total]));
}

export async function policyPool(game, history, table, config, infer) {
  let source = 'table';
  const counts = table[positionKey(game)];
  let dist;
  const maia = async () => {
    const logits = await infer(tokensFromHistory(history,config));
    const probabilities = legalProbabilities(game,logits);
    return normalize(Object.fromEntries(Object.entries(probabilities).map(([move,p]) => [move,p**(1/config.temperature)])));
  };
  if (counts && Object.values(counts).reduce((a,b) => a+b,0) >= config.tableMinCount) dist = normalize(counts);
  else { source = 'maia'; dist = await maia(); }
  let pool = playable(game,dist);
  if (!Object.keys(pool).length && source === 'table') {
    source = 'maia'; dist = await maia(); pool = playable(game,dist);
  }
  if (!Object.keys(pool).length) {
    pool = playable(game,Object.fromEntries(game.moves({verbose:true}).map(m => [toUci(m),1])));
    if (!Object.keys(pool).length) pool = dist;
  }
  const cutoff = config.minP * Math.max(...Object.values(pool));
  return {source, probabilities:normalize(Object.fromEntries(Object.entries(pool).filter(([,p]) => p >= cutoff)))};
}

export function sample(probabilities, random=Math.random) {
  let r = random();
  const moves = Object.keys(probabilities);
  for (const move of moves) { r -= probabilities[move]; if (r <= 0) return move; }
  return moves[moves.length-1];
}

export function restoreGame(initialFen, moves) {
  const game = new Chess(initialFen);
  const history = [game.fen()];
  for (const uci of moves) { game.move(moveObject(uci)); history.push(game.fen()); }
  return {game,history};
}
