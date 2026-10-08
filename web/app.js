let state, token, selected = null, busy = false;
const $ = id => document.getElementById(id);
let replayStep = null;
const ACTION_PAUSE_MS = 1500;
let recommendation = null, hintLoading = false, hintMessage = '';
const playerWind = p => windNames[(state?.seats?.[p] || p+1)-1];
async function loadHint(){
  if(!state?.started || state.over || state.decision!=='turn')return;
  const revision=state.revision;
  hintLoading=true;renderHint();
  try{
    const response=await request('/api/hint',{revision});
    if(state.revision!==revision||busy)return;
    recommendation=response.recommendation;hintMessage='';
  }catch(error){if(state.revision===revision)hintMessage='Recommendation unavailable. You can still play.';}
  finally{if(state.revision===revision&&!busy){hintLoading=false;renderHint();}}
}
function renderHint(){
  const box=$('recommendation');box.replaceChildren();
  document.querySelectorAll('#hand .recommended').forEach(node=>node.classList.remove('recommended'));
  if(busy||!state?.started||state.over||state.decision!=='turn')return;
  if(hintLoading){box.textContent='Calculating recommended discard and estimated win chance…';return;}
  if(!recommendation){box.textContent=hintMessage;return;}
  box.append(tile(recommendation.tile,true),el('span',`Recommended: ${label(recommendation.tile)} · Estimated hand win chance: ${recommendation.win_percent.toFixed(1)}%`));
  document.querySelectorAll('#hand .tile').forEach(node=>{if(node.classList.contains(`face-${recommendation.tile[0]}-${recommendation.tile[1]}`))node.classList.add('recommended');});
}
const names = {east:'東',south:'南',west:'西',north:'北',red:'中',green:'發',white:'白'};
const label = t => t[0]==='wind' ? `${windLabels[t[1]]} wind` : `${t[1]} ${t[0]}`;
const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
const seats = ['right','top','left'];
const bots = ['Jade','Bamboo','Lotus'];
const windNames = ['東 Dong', '南 Nan', '西 Xi', '北 Bei'];
const windLabels = {east:'Dong', south:'Nan', west:'Xi', north:'Bei'};
function face(t) {
  if(t[0]==='dot') return ['','●','●●','∴','∷','⁙','∷∶','∷∴','∷∷','⁙∷'][t[1]];
  if(t[0]==='bamboo') return `${t[1]}竹`;
  if(t[0]==='wan') return `${['','一','二','三','四','五','六','七','八','九'][t[1]]}萬`;
  return names[t[1]] || ({flower:'✿',season:'❋',animal:'♧'}[t[0]] || String(t[1]));
}
function el(tag, text, cls) { const node=document.createElement(tag); if(text!==undefined) node.textContent=text; if(cls) node.className=cls; return node; }
function tile(t, mini=false, action=null) {
  const node=el(action===null?'span':'button',undefined,`tile ${t[0]}${mini?' mini':''}`);
  node.classList.add('photo-tile',`face-${t[0]}-${t[1]}`);
  const artwork=el('span',undefined,'tile-art');
  artwork.setAttribute('aria-hidden','true');
  node.append(artwork);
  node.title=label(t); node.setAttribute('aria-label',label(t));
  if(action!==null){node.disabled=busy;node.title=`Discard ${label(t)}`;node.setAttribute('aria-label',`Discard ${label(t)}`);node.onclick=()=>update('/api/action',{id:action,revision:state.revision});}
  return node;
}
function publicTiles(p) {
  const box=el('div',undefined,'public-tiles');
  for(const [title,groups] of [['Melds',state.melds[p]],['Bonus',[state.bonus[p]]]]) {
    const section=el('div',undefined,`public-group ${title.toLowerCase()}`);
    section.append(el('p',title,'public-label'));
    if(!groups.flat().length) section.append(el('span','—','empty'));
    for(const group of groups){const row=el('span',undefined,'meld');group.forEach(t=>row.append(tile(t,true)));section.append(row);}
    box.append(section);
  }
  return box;
}
function renderDiscards(){
  const river=$('discard-river');river.replaceChildren();
  for(const p of [0,1,2,3]){
    const row=el('section',undefined,'river-row');
    row.append(el('span',`${playerWind(p)} · ${state.discards[p].length}`,'river-label'));
    const pile=el('div',undefined,'river-pile');
    state.discards[p].forEach((t,i)=>{
      const face=tile(t,true);
      if(state.last_discard?.player===p&&i===state.discards[p].length-1)face.classList.add('latest-discard');
      pile.append(face);
    });
    row.append(pile);river.append(row);
  }
  fitDiscards();
}
function fitDiscards(){
  const river=$('discard-river');
  if(!river||!state?.started)return;
  const width=Math.max(1,river.clientWidth-88);
  const count=Math.max(1,...state.discards.map(p=>p.length));
  // All four piles remain in view; no clipping, scrolling, or removed tiles.
  const size=Math.min(36,Math.max(1,(width-2*(count-1))/count));
  river.style.setProperty('--river-tile-width',`${size}px`);
  river.style.setProperty('--river-tile-height',`${size*1.42}px`);
}
new ResizeObserver(fitDiscards).observe($('discard-river'));
async function request(path, data) {
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Table-Token':token},body:JSON.stringify(data)});
  const body=await response.json(); if(!response.ok) throw new Error(body.error || 'Request failed');
  if(body.token) token=body.token; return body;
}
async function update(path,data){
  if(busy)return; busy=true; recommendation=null;hintLoading=false;hintMessage=''; $('error').textContent=''; render();
  try {
    const next=await request(path,data);selected=null;
    for(const step of next.playback || []){
      replayStep=step;state=step.state;render();
      if(step.action.kind==='discard') animateDiscard(step.player);
      await new Promise(resolve=>setTimeout(resolve,ACTION_PAUSE_MS));
    }
    state=next;
  }
  catch(error){$('error').textContent=error.message;try{state=await request('/api/state');selected=null;}catch{}}
  finally{replayStep=null;busy=false;render();loadHint();}
}
function animateDiscard(player){
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const target=$('pending').querySelector('.tile');
  const origin=player===0?$('hand'):document.querySelector(`.seat-${seats[player-1]} .backs`);
  if(!target||!origin)return;
  const from=origin.getBoundingClientRect(),to=target.getBoundingClientRect();
  const x=from.left+from.width/2-to.left-to.width/2;
  const y=from.top+from.height/2-to.top-to.height/2;
  target.animate([{transform:`translate(${x}px,${y}px) rotate(-12deg)`,opacity:.6},{transform:'translate(0,0) rotate(0)',opacity:1}],{duration:550,easing:'cubic-bezier(.2,.7,.3,1)'});
}
function actionButton(text,id){const b=el('button',text);b.disabled=busy;b.onclick=()=>update('/api/action',{id,revision:state.revision});return b;}
function render(){
  $('new').disabled=busy;$('new').textContent=busy?'Playing…':'New hand';
  if(!state?.started){
    $('opponents').replaceChildren();
    for(const p of [1,2,3]){
      const box=el('section',undefined,`player seat-${seats[p-1]}`);
      box.append(el('div',bots[p-1][0],'avatar'),el('h3',`${bots[p-1]} · AI`),el('p',`Player ${p+1} · ${windNames[p]}`,'bot-info'));
      const backs=el('div',undefined,'backs');for(let i=0;i<13;i++)backs.append(el('span',undefined,'back'));
      box.append(backs);$('opponents').append(box);
    }
    return;
  }
  $('opponents').replaceChildren();
  for(const p of [1,2,3]){
    const box=el('section',undefined,`player seat-${seats[p-1]}`);
    box.classList.toggle('offering',replayStep?replayStep.player===p:state.source===p);
    box.append(el('div',bots[p-1][0],'avatar'),el('h3',`${bots[p-1]} · AI`),el('p',`Player ${p+1} · ${windNames[state.seats[p]-1]}`,'bot-info'));
    const backs=el('div',undefined,'backs');backs.setAttribute('aria-label',`${state.counts[p]} concealed tiles`);
    if(state.winning_hand?.player===p){
      backs.classList.add('revealed-hand');backs.setAttribute('aria-label','Winning hand');
      state.winning_hand.hand.forEach(t=>backs.append(tile(t,true)));
    }else for(let i=0;i<state.counts[p];i++)backs.append(el('span',undefined,'back'));
    box.append(backs,publicTiles(p));$('opponents').append(box);
  }
  $('meta').textContent=`${windLabels[state.wind]} round · ${state.wall} drawable tiles · Turn ${state.turn} · Dong → Nan → Xi → Bei`;
  $('your-seat').textContent=`Player 1 · ${windNames[state.seats[0]-1]}`;
  renderDiscards();
  $('status').textContent=busy?'The table is playing…':state.over?(state.result.winner===null?'The hand is a draw':state.result.winner===0?'You win!':`Player ${state.result.winner+1} wins`):state.decision==='turn'?'Your turn':state.decision==='rob_kong'?'A chance to rob the Kong':'Claim the discard?';
  if(replayStep)$('status').textContent=`${playerWind(replayStep.player)} · ${replayStep.action.kind==='discard'?'discards':replayStep.action.kind.toUpperCase()}`;
  $('pending').replaceChildren();
  const thrown=replayStep?.action.kind==='discard'?{player:replayStep.player,tile:replayStep.action.tile}:state.pending?{player:state.source,tile:state.pending}:state.last_discard;
  if(thrown){$('pending').append(el('span',`${playerWind(thrown.player)} ${state.pending?'offers':'discarded'} `),tile(thrown.tile));}
  $('hand').replaceChildren();
  for(const t of state.hand){const action=state.actions.find(a=>a.kind==='discard'&&same(a.tile,t));$('hand').append(tile(t,false,action?.id??null));}
  $('instruction').textContent=state.over?'Start a new hand to play again.':state.decision==='turn'?'Click a tile once to discard it.':'Choose a claim or pass.';
  if(busy)$('instruction').textContent='Watch the table — actions pause for 1.5 seconds.';
  $('own-public').replaceChildren(publicTiles(0));$('actions').replaceChildren();
  for(const a of state.actions.filter(a=>a.kind!=='discard')){
    let text=a.kind.toUpperCase();if(a.tile)text+=` · ${label(a.tile)}`;if(a.chi)text+=` with ${a.chi.map(label).join(' + ')}`;
    $('actions').append(actionButton(text,a.id));
  }
  $('winner-panel').replaceChildren();$('winner-panel').hidden=!state.winning_hand;
  if(state.winning_hand){
    const w=state.winning_hand;
    $('winner-panel').append(el('h2',`${playerWind(w.player)} · Player ${w.player+1} wins — ${state.result.tai} tai`));
    $('winner-panel').append(el('p',state.next_dealer===state.dealer?`Dong stays with Player ${state.dealer+1}.`:`Next hand: Player ${state.next_dealer+1} becomes Dong; all seat winds move one place.`));
    for(const [name,groups] of [['Winning hand (including winning tile)',[w.hand]],['Exposed melds',w.melds],['Bonus tiles',[w.bonus]]]){
      if(!groups.flat().length)continue;
      const section=el('div',undefined,'winning-group');section.append(el('h3',name));
      for(const group of groups){const rack=el('div',undefined,'winning-rack');group.forEach(t=>rack.append(tile(t)));section.append(rack);}
      $('winner-panel').append(section);
    }
  }
  $('result').replaceChildren();
  if(state.result){const r=state.result;$('result').append(el('p',r.winner===null?'No winner this hand.':`${r.tai} tai · ${r.self_drawn?'Self-drawn':`Discard from Player ${r.shooter+1}`}`));for(let p=0;p<4;p++)$('result').append(el('span',`${p===0?'You':`Player ${p+1}`}: ${r.total[p]>0?'+':''}${r.total[p]} units`,'score'));}
  $('history').replaceChildren(...[...state.events].reverse().map(line=>el('li',line.replace(/\b(east|south|west|north)\b/gi,word=>windLabels[word.toLowerCase()]))));
  renderHint();
}
$('new').onclick=()=>{if(state?.started&&!state.over&&!confirm('End this hand and deal a new one?'))return;update('/api/new',{});};
update('/api/state');

