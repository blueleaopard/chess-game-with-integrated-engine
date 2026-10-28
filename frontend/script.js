const fileInput=document.getElementById('pgnFile');
const dropzone=document.getElementById('dropzone');
const fileTitle=document.getElementById('fileTitle');
const fileSub=document.getElementById('fileSub');
const playerName=document.getElementById('playerName');
const modelName=document.getElementById('modelName');
const errorBox=document.getElementById('error');
const processButton=document.getElementById('processButton');
const progressArea=document.getElementById('progressArea');
let selectedFile=null;
let pollTimer=null;

function selectFile(file){
  if(!file)return;
  if(!file.name.toLowerCase().endsWith('.pgn')){
    errorBox.textContent='Please choose a .pgn file.';
    return;
  }
  selectedFile=file;
  errorBox.textContent='';
  fileTitle.textContent=file.name;
  fileSub.textContent=(file.size/1048576).toFixed(2)+' MB · Ready to process';
}
fileInput.addEventListener('change',()=>selectFile(fileInput.files[0]));
dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('over')});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('over'));
dropzone.addEventListener('drop',e=>{
  e.preventDefault();dropzone.classList.remove('over');
  selectFile(e.dataTransfer.files[0]);
});

function setProgress(task){
  const percent=Math.max(0,Math.min(100,Number(task.percent)||0));
  progressArea.hidden=false;
  document.getElementById('progressMessage').textContent=task.message||task.stage||'Working…';
  document.getElementById('progressPercent').textContent=Math.round(percent)+'%';
  document.getElementById('progressFill').style.width=percent+'%';
  const logOutput=document.getElementById('logOutput');
  logOutput.textContent=(task.logs||[]).slice(-16).join('\n');
  logOutput.scrollTop=logOutput.scrollHeight;
  const stages=[
    ['stage1',2,25],['stage2',25,27],['stage3',27,68],['stage4',68,96],['stage5',96,100]
  ];
  stages.forEach(([id,start,end])=>{
    const el=document.getElementById(id);
    if(!el)return;
    el.classList.toggle('complete',percent>=end);
    el.classList.toggle('current',percent>=start&&percent<end);
  });
}
function showMetrics(task){
  const m=task.metrics||{};
  document.getElementById('summaryStatus').textContent=task.status==='completed'?'MODEL READY':'TRAINING FAILED';
  const gameCount=task.stats?.player_games ?? (Number(m.training_games||0)+Number(m.test_games||0));
  document.getElementById('metricGames').textContent=gameCount?Number(gameCount).toLocaleString():'—';
  document.getElementById('metricPositions').textContent=(Number(m.training_positions||0)+Number(m.test_positions||0)).toLocaleString()||'—';
  document.getElementById('metricCandidates').textContent=Number(m.candidate_rows||0).toLocaleString()||'—';
  document.getElementById('metricTop1').textContent=typeof m.top1==='number'?(m.top1*100).toFixed(2)+'%':'—';
  document.getElementById('summaryFootnote').textContent=task.status==='completed'
    ?`Saved model: ${task.model?.model_name||'Player model'} · Top-3: ${typeof m.top3==='number'?(m.top3*100).toFixed(2)+'%':'—'} · MRR: ${typeof m.mrr==='number'?m.mrr.toFixed(4):'—'}`
    :(task.error||task.message||'Training did not complete.');
}
async function pollProgress(taskId){
  try{
    const response=await fetch(`/api/progress/${taskId}`);
    const task=await response.json();
    if(!response.ok)throw new Error(task.error||'Could not read progress.');
    setProgress(task);
    if(task.status==='completed'){
      clearInterval(pollTimer);pollTimer=null;
      processButton.disabled=false;processButton.innerHTML='Build another model <b>→</b>';
      showMetrics(task);
      await loadModels();
    }else if(task.status==='failed'){
      clearInterval(pollTimer);pollTimer=null;
      processButton.disabled=false;processButton.innerHTML='Try again <b>→</b>';
      showMetrics(task);
      errorBox.textContent=task.error||'Training failed. See the log output.';
    }
  }catch(error){
    errorBox.textContent='Lost connection to backend: '+error.message;
  }
}
processButton.addEventListener('click',async()=>{
  errorBox.textContent='';
  if(!selectedFile){errorBox.textContent='Choose a PGN file first.';return;}
  if(!playerName.value.trim()){errorBox.textContent='Enter the target player’s name.';playerName.focus();return;}
  const form=new FormData();
  form.append('pgn',selectedFile);
  form.append('player_name',playerName.value.trim());
  form.append('model_name',modelName.value.trim());
  processButton.disabled=true;
  processButton.textContent='Starting…';
  progressArea.hidden=false;
  document.getElementById('logOutput').textContent='';
  try{
    const response=await fetch('/api/train',{method:'POST',body:form});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'Could not start training.');
    processButton.textContent='Training…';
    if(pollTimer)clearInterval(pollTimer);
    pollTimer=setInterval(()=>pollProgress(data.task_id),1500);
    await pollProgress(data.task_id);
  }catch(error){
    processButton.disabled=false;
    processButton.innerHTML='Process games &amp; build model <b>→</b>';
    errorBox.textContent=error.message;
  }
});
async function loadModels(){
  try{
    const response=await fetch('/api/models');
    const data=await response.json();
    const names=(data.models||[]).map(m=>`${m.model_name} (${m.target_player})`);
    document.getElementById('summaryFootnote').textContent += names.length?` · Saved models: ${names.join(', ')}`:'';
  }catch(_error){}
}
fetch('/api/status').then(r=>r.json()).then(data=>{
  if(data.status!=='running')throw new Error('Backend not running');
}).catch(()=>{
  errorBox.textContent='Backend is not connected. Start it with: python app.py';
});
loadModels();
