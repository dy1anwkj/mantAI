/* Physical table presentation. All dice, slots and draws come from Python. */
let openingScene = false, gapVisible = false;
const pauseScene = ms => new Promise(resolve=>setTimeout(resolve,ms));
const slotKey = slot => slot.join(':');
function stackElement(slot){
  return slot && document.querySelector(`#field-walls [data-player="${slot[0]}"][data-stack="${slot[1]}"]`);
}
function renderField(){
  const root=$('field-walls');root.replaceChildren();
  if(!state?.setup)return;
  const placement=['bottom','right','top','left'];
  for(let p=0;p<4;p++){
    const wall=el('div',undefined,`field-wall wall-${placement[p]}`);
    wall.setAttribute('aria-label',`${playerWind(p)} wall; stacks counted from owner's right`);
    state.wall_stacks[p].forEach((count,i)=>{
      const stack=el('span',undefined,`field-stack count-${count}`);
      stack.dataset.player=p;stack.dataset.stack=i+1;
      stack.title=`${playerWind(p)}: stack ${i+1} from right, ${count} tiles`;
      const sameStack=slot=>slot && slot[0]===p && slot[1]===i+1;
      if(state.wall_ends && (!openingScene||gapVisible)){
        if(sameStack(state.wall_ends.live)){stack.classList.add('live-end');stack.title+=' · live draw / dealing end';}
        if(sameStack(state.wall_ends.replacement)){stack.classList.add('replacement-end');stack.title+=' · flower / Kong replacement end';}
      }
      if(openingScene)stack.append(el('small',String(i+1)));
      wall.append(stack);
      if((!openingScene||gapVisible)&&p===state.setup.selected&&i+1===state.setup.gap_after){
        const gap=el('span',undefined,'physical-gap');gap.title='Original wall break';wall.append(gap);
      }
    });
    root.append(wall);
  }
}
function receivingRack(p){return p===0?$('hand'):document.querySelector(`.seat-${seats[p-1]} .backs`);}
async function flySlot(slot,player,replacement=false){
  const source=stackElement(slot),destination=receivingRack(player);
  if(!source||!destination)return;
  const start=source.getBoundingClientRect(),end=destination.getBoundingClientRect();
  const chip=el('span',undefined,`flying-tile${replacement?' replacement-flight':''}`);
  chip.style.left=`${start.left}px`;chip.style.top=`${start.top}px`;document.body.append(chip);
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  try{await chip.animate([{transform:'translate(0,0) rotate(0)'},{transform:`translate(${end.left+end.width/2-start.left}px,${end.top+end.height/2-start.top}px) rotate(12deg)`}],{duration:reduce?1:360,easing:'ease-in-out',fill:'forwards'}).finished;}finally{chip.remove();}
}
async function animateWallDraws(step){
  for(const draw of step.state.wall_draws||[]){
    if(draw.player===null)continue;
    if(draw.end==='replacement')$('status').textContent=`${playerWind(draw.player)} draws from the flower / Kong replacement end`;
    await flySlot(draw.slot,draw.player,draw.end==='replacement');
  }
}
async function showSetup(next){
  const setup=next.setup;
  openingScene=true;gapVisible=false;
  document.body.classList.add('opening-scene');
  state={...next,points:next.hand_start_points,hand:[],counts:[0,0,0,0],discards:[[],[],[],[]],melds:[[],[],[],[]],bonus:[[],[],[],[]],
    actions:[],events:[],result:null,winning_hand:null,over:false,last_discard:null,pending:null,source:null,
    wall_stacks:setup.stacks.map(n=>Array(n).fill(2)),wall_ends:null,turn:0,wall:133};
  render();$('opening').hidden=false;
  $('status').textContent='Banker throws three dice';$('opening-message').textContent='Rolling…';
  $('opening-counts').textContent='';
  const dice=$('throw-dice');dice.replaceChildren();
  const source=receivingRack(setup.banker).getBoundingClientRect();
  for(let i=0;i<3;i++){
    const die=el('span','⚀','thrown-die');die.setAttribute('aria-label','Rolling die');dice.append(die);
    const end=die.getBoundingClientRect();
    if(!matchMedia('(prefers-reduced-motion: reduce)').matches)die.animate([
      {transform:`translate(${source.left+source.width/2-end.left}px,${source.top-end.top}px) rotate(-420deg) scale(.6)`},
      {transform:`translate(${25-i*20}px,-30px) rotate(35deg) scale(1.15)`,offset:.65},
      {transform:'translate(0,0) rotate(0) scale(1)'}],{duration:1200+i*170,easing:'ease-out'});
  }
  // Flickering faces are cosmetic; the settled values are the backend roll.
  const flicker=setInterval(()=>[...dice.children].forEach(d=>{d.textContent=['⚀','⚁','⚂','⚃','⚄','⚅'][Math.floor(Math.random()*6)];}),90);
  await pauseScene(1650);clearInterval(flicker);
  [...dice.children].forEach((d,i)=>{d.textContent=['⚀','⚁','⚂','⚃','⚄','⚅'][setup.dice[i]-1];d.setAttribute('aria-label',`Die ${i+1}: ${setup.dice[i]}`);});
  $('opening-message').textContent=`${setup.dice.join(' + ')} = ${setup.total}. Counting anticlockwise from the banker as 1…`;
  for(let n=1;n<=setup.total;n++){
    document.querySelectorAll('.counting-seat').forEach(e=>e.classList.remove('counting-seat'));
    const p=(setup.banker+n-1)%4;
    (p===0?document.querySelector('.your-area'):document.querySelector(`.seat-${seats[p-1]}`)).classList.add('counting-seat');
    $('opening-counts').textContent=`Count ${n}: ${playerWind(p)}`;
    await pauseScene(100);
  }
  $('opening-message').textContent=`${playerWind(setup.selected)} wall selected. Counting ${setup.total} stacks from that player's right toward their left.`;
  for(let n=1;n<=setup.total;n++){
    stackElement([setup.selected,n,0]).classList.add('counted-stack');await pauseScene(90);
  }
  gapVisible=true;
  state.wall_ends={live:setup.live_start,replacement:setup.replacement_start};renderField();
  $('opening-message').textContent=`Break after stack ${setup.total}. Gold end: deal and normal draws. Pink end: flower and Kong replacements.`;
  await pauseScene(1700);
  // Ring order follows each owner's right-to-left direction around corners.
  const ring=[];for(const p of [0,3,2,1])for(let n=1;n<=setup.stacks[p];n++)for(const z of [0,1])ring.push([p,n,z]);
  const cut=ring.findIndex(s=>slotKey(s)===slotKey(setup.live_start));
  let remaining=ring.slice(cut).concat(ring.slice(0,cut));
  function consume(slots){
    const removed=new Set(slots.map(slotKey));
    slots.forEach(([p,n])=>state.wall_stacks[p][n-1]--);
    remaining=remaining.filter(s=>!removed.has(slotKey(s)));
    state.wall_ends={live:remaining[0],replacement:remaining[remaining.length-1]};renderField();
  }
  function updateHands(){
    for(let p=0;p<4;p++){
      const rack=receivingRack(p);rack.replaceChildren();
      for(let n=0;n<state.counts[p];n++)rack.append(el('span',undefined,'back'));
    }
    $('opening-counts').textContent=setup.order.map(p=>`${playerWind(p)}: ${state.counts[p]} tiles`).join(' · ');
  }
  for(const packet of setup.packets){
    $('opening-message').textContent=`Dealing ${packet.count} tiles to ${playerWind(packet.player)} from the gold end.`;
    await Promise.all(packet.slots.map((slot,i)=>pauseScene(i*45).then(()=>flySlot(slot,packet.player))));
    consume(packet.slots);state.counts[packet.player]+=packet.count;updateHands();await pauseScene(100);
  }
  for(const draw of setup.replacements||[]){
    $('opening-message').textContent=`${playerWind(draw.player)} exposes a bonus tile and draws its replacement from the pink end.`;
    if(draw.exposed_bonus){
      state.bonus[draw.player].push(draw.exposed_bonus);
      const owner=draw.player===0?$('own-public'):document.querySelector(`.seat-${seats[draw.player-1]}`);
      const current=owner.querySelector('.public-tiles');
      if(current)current.replaceWith(publicTiles(draw.player));
    }
    await flySlot(draw.slot,draw.player,true);consume([draw.slot]);await pauseScene(100);
  }
  $('opening-message').textContent='Dealing complete. Banker discards first. Discards land inside the four walls.';
  await pauseScene(800);
  document.querySelectorAll('.counting-seat').forEach(e=>e.classList.remove('counting-seat'));
  openingScene=false;document.body.classList.remove('opening-scene');$('opening').hidden=true;
}
