"use strict";

const $ = (q, root = document) => root.querySelector(q);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const token = $('meta[name="scholarloop-token"]').content;
const state = {boot:null, run:null, detail:null, tab:"overview", job:null, offset:0, log:"", filter:"", polling:false, selectedFile:"", navigation:0, artifactRequest:0};
const tabs = [
  ["overview","Overview","grid"], ["research","Literature & Directions","book"], ["ideas","Research Ideas","bulb"],
  ["experiments","Experiments","flask"], ["review","Scientific Review","shield"], ["writing","Writing","pen"],
  ["files","Artifacts","folder"], ["tools","Evaluation & Tools","sliders"]
];
const paths = {
  grid:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  book:'<path d="M12 5v16M3 4c4-1 6 0 9 2 3-2 5-3 9-2v15c-4-1-6 0-9 2-3-2-5-3-9-2z"/>',
  bulb:'<path d="M9 18h6m-5 3h4M8 14a7 7 0 1 1 8 0c-1 1-1 2-1 2H9s0-1-1-2z"/>',
  flask:'<path d="M9 3h6m-5 0v6l-6 10q-1 2 2 2h12q3 0 2-2L14 9V3M7 15h10"/><path d="m10 18 .1 .1m4-1 .1 .1"/>',
  shield:'<path d="m12 3 8 3v6c0 4-4 7-8 9-4-2-8-5-8-9V6z"/><path d="m8 12 3 3 5-6"/>',
  pen:'<path d="m16 3 5 5-12 12-6 1 1-6zM13 6l5 5M3 21h18"/>',
  folder:'<path d="M3 7V5a2 2 0 0 1 2-2h5l3 3h6a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM3 9h18"/>',
  sliders:'<path d="M5 3v6m0 4v8M12 3v11m0 4v3M19 3v3m0 4v11M2 9h6m1 5h6m1-8h6"/>',
  plus:'<path d="M12 5v14M5 12h14"/>', arrow:'<path d="M5 12h14m-5-5 5 5-5 5"/>',
  play:'<path d="m8 4 12 8-12 8z"/>', check:'<path d="m5 12 4 4L19 6"/>',
  file:'<path d="M14 2H5v20h14V7zM14 2v6h5M8 13h8m-8 4h6"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>', download:'<path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/>',
  clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  link:'<path d="m10 13 4-4m-5 7-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m0 10a4 4 0 0 0 6 0l5-5a4 4 0 0 0-6-6l-2 2"/>',
  menu:'<path d="M4 6h16M4 12h16M4 18h16"/>', dots:'<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  stop:'<rect x="5" y="5" width="14" height="14" rx="2"/>', refresh:'<path d="M20 7a9 9 0 1 0 1 9M20 3v5h-5"/>',
  leaf:'<path d="M20 3C7 2 2 8 5 16c7 7 15-1 15-13zM3 21 15 9"/>'
};
const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
const tag = (text, color="") => `<span class="tag ${color}">${esc(text)}</span>`;
function actionReason(action) {
  const d=state.detail;
  if(!d)return "Open a research project first.";
  if(d.config.readonly)return "This archive is read-only.";
  if(busy())return "Wait for the current task to finish, or stop it first.";
  if(d.config.mode==="demo"&&!["demo","evaluate","traces","decision_bench"].includes(action))return "Create a Live research project to use this action.";
  if(["directions","ideas","inject"].includes(action)&&!fileExists("lit_review.md"))return "Complete the literature review first.";
  if(["select","refine_ideas","pilot_plan"].includes(action)&&!d.session.ideas?.length)return "Generate candidate ideas first.";
  if(action==="campaign"&&!Object.keys(d.handoff||{}).length)return "Select a research idea first.";
  if(action.startsWith("write")&&(d.review.status!=="READY_TO_WRITE"||d.review.release_error))return "Complete scientific review and approve the current evidence before writing.";
  return "";
}
function button(action, label, style="", preset={}) {
  const reason=actionReason(action);
  return `<button type="button" class="btn ${style}" data-action="${esc(action)}" data-preset="${esc(JSON.stringify(preset))}" ${reason?`disabled title="${esc(reason)}"`:""}>${esc(label)}</button>`;
}
const artifactButton = (path, label="View", style="small") => `<button type="button" class="btn ${style}" data-file="${esc(path)}">${esc(label)} ${icon("arrow")}</button>`;
const timeText = t => new Date(t * 1000).toLocaleString("en-US",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
const statusLabels = {running:"Running",succeeded:"Completed",attention:"Needs attention",failed:"Failed",cancelled:"Stopped",interrupted:"Interrupted"};
const statusColors = {running:"blue",succeeded:"green",attention:"amber",failed:"red",cancelled:"amber",interrupted:"amber"};
const reviewLabels = {READY_FOR_AUTHOR:"Awaiting author approval",READY_TO_WRITE:"Approved for writing",NEEDS_EXPERIMENTS:"More experiments needed",REVIEWING:"Under review",REVIEW_FAILED:"Review incomplete",NEEDS_REVISION:"Revision needed"};

async function api(path, data) {
  const res = await fetch(path, data === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json","X-ScholarLoop-Token":token},body:JSON.stringify(data)});
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}
let toastTimer;
function toast(message, error=false) {
  const el = $("#toast"); el.textContent = message; el.className = error ? "error" : ""; el.hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,error?7000:3500);
}
function busy() { return state.detail?.jobs.some(j=>j.status==="running") || false; }
function fileExists(name) {
  if(["writing.json","outline.md","key_points.md","paper.md","paper_traced.md","review.md","evidence_map.md","approved_evidence.json"].includes(name))return !!state.detail?.writing.artifacts?.[name];
  return state.detail?.artifacts.some(f=>f.path===name);
}
function profile() { return state.boot.profiles.find(p=>p.id===state.detail?.config.profile); }
// An archive that kept a stage only as a document: point to the document, never to an action.
function recorded(file, what, label) {
  return fileExists(file) ? `<div class="empty"><div class="empty-symbol">${icon("folder")}</div><h3>${esc(what)}</h3><p>This archived run did not record this stage in a structured form.</p>${artifactButton(file,label)}</div>`
    : `<div class="empty"><div class="empty-symbol">${icon("folder")}</div><h3>Nothing was recorded for this stage</h3><p>This archived run kept no record here.</p></div>`;
}
// A cited source as a link to its public page, when the identifier is one we recognise.
function sourceLink(id) {
  const text=String(id??"").trim();
  let m;
  if((m=/^arxiv:(\d{4}\.\d{4,5}(?:v\d+)?|[a-z.-]+\/\d{7}(?:v\d+)?)$/i.exec(text)))return "https://arxiv.org/abs/"+m[1];
  if((m=/^doi:(10\.\d{4,9}\/[^\s"<>]+)$/i.exec(text)))return "https://doi.org/"+encodeURI(m[1]);
  if((m=/^openalex:(W\d+)$/i.exec(text)))return "https://openalex.org/"+m[1];
  return "";
}
function sourceChip(id) {
  const url=sourceLink(id);
  return url?`<a class="source-chip" href="${esc(url)}" target="_blank" rel="noopener noreferrer" title="Open ${esc(id)} in a new tab">${esc(id)} ${icon("arrow")}</a>`:`<span>${esc(id)}</span>`;
}
// One direction, in the same shape as an idea: its open question is the point of the card.
function directionCard(d, i) {
  const chosen=state.detail.session.chosen_directions?.includes(d.id), n=String(i+1).padStart(2,"0");
  const stance={develops:"Develops your hypothesis",challenges:"Challenges your hypothesis",outside:"A new perspective",author:"Your own direction"}[d.stance]||d.stance||d.origin||"";
  return `<article class="panel idea-card${chosen?" chosen":""}"><header class="idea-head"><div class="idea-rank">${n}</div><div class="grow"><div class="idea-kicker">${esc([`Direction ${n}`,stance].filter(Boolean).join(" · "))}</div><h3>${esc(d.name)}</h3></div><div class="idea-tags">${chosen?tag("Chosen","green"):""}</div></header>`+
    `<section class="idea-hypothesis"><div class="idea-label">Open question</div><p>${esc(d.open_question??"")}</p></section>`+
    `<details class="idea-fold" open><summary>Thesis</summary>${points(d.thesis)}</details>`+(d.first_experiment?`<details class="idea-fold"><summary>First experiment</summary>${points(d.first_experiment)}</details>`:"")+
    `<footer class="idea-foot"><div class="idea-sources"></div>${button("ideas","Generate ideas in this direction","",{directions:String(i+1)})}</footer></article>`;
}
// One idea: the hypothesis leads, scores and tournament standing sit beside the title, and the
// longer parts fold away. The chosen idea is marked and starts unfolded.
function ideaCard(idea, i) {
  const chosen=state.detail.session.selected===i+1, t=idea.tournament||{}, elo=t.elo??t.rating, h=idea.hypothesis||{};
  const dots=n=>`<span class="idea-dots" aria-hidden="true">${[1,2,3,4,5].map(k=>`<i${k<=Math.round(n)?' class="on"':""}></i>`).join("")}</span>`;
  const score=(label,n)=>Number.isFinite(Number(n))&&n!==null&&n!==""?`<span class="idea-score">${label} ${dots(Number(n))}<b>${Math.round(Number(n))}/5</b></span>`:"";
  const novelty=idea.novelty_review?.status;
  const fold=(title,text,open)=>text?`<details class="idea-fold"${open?" open":""}><summary>${title}</summary>${points(text)}</details>`:"";
  const kicker=[`Idea ${String(i+1).padStart(2,"0")}`,t.rank?`Rank ${t.rank} in the tournament`:"",idea.strategy==="human"?"Added by the author":""].filter(Boolean).join(" · ");
  return `<article class="panel idea-card${chosen?" chosen":""}"><header class="idea-head"><div class="idea-rank">${String(i+1).padStart(2,"0")}</div><div class="grow"><div class="idea-kicker">${esc(kicker)}</div><h3>${esc(idea.title)}</h3></div><div class="idea-tags">${chosen?tag("Selected","green"):""}${elo?tag(`Elo ${Math.round(elo)}`,"outline"):chosen?"":tag(idea.status||"candidate","outline")}${t.wins!==undefined?`<small>${t.wins}W · ${t.losses}L · ${t.draws}D</small>`:""}</div></header>`+
    `<section class="idea-hypothesis"><div class="idea-label">Hypothesis</div><p>${esc(h.claim??"")}</p>${h.predicted_effect?`<div class="idea-label">Predicted effect</div>${points(h.predicted_effect)}`:""}</section>`+
    `<div class="idea-scores">${score("Feasibility",idea.feasibility)}${score("Impact",idea.impact)}${novelty?`<span class="idea-score">Novelty check <b>${esc(String(novelty).replaceAll("_"," "))}</b></span>`:""}</div>`+
    fold("Approach",idea.description,chosen)+fold("Why it might work",idea.design?.mechanism,false)+fold("Experiment plan",idea.experiment_plan,chosen)+
    `<footer class="idea-foot"><div class="idea-sources">${(idea.sources||[]).map(sourceChip).join("")}<span class="mono">${esc(idea.idea_id)}</span></div>${button("select",chosen?"Confirm selection again":"Select this idea",chosen?"soft":"",{select:i+1})}</footer></article>`;
}
// A long recorded paragraph, shown one sentence per point. The words are not changed.
function sentences(text) {
  return String(text??"").split(/(?<=[.!?])\s+(?=[A-Z"'(\[])/).reduce((out,part)=>{
    const last=out[out.length-1];
    if(last&&/(?:^|\s)(?:e\.g|i\.e|cf|vs|etc|Fig|Eq|No)\.$/.test(last))out[out.length-1]=last+" "+part; else out.push(part);
    return out;
  },[]).filter(Boolean);
}
function points(text) {
  const parts=sentences(text);
  return parts.length<2?`<p>${esc(text??"")}</p>`:`<ul class="points">${parts.map(part=>`<li>${esc(part)}</li>`).join("")}</ul>`;
}
// The first few points, with the rest one click away.
function leadPoints(text, keep=3) {
  const parts=sentences(text), list=items=>`<ul class="points">${items.map(part=>`<li>${esc(part)}</li>`).join("")}</ul>`;
  if(parts.length<=keep+1)return points(text);
  return list(parts.slice(0,keep))+`<details class="more-points"><summary>${parts.length-keep} more points</summary>${list(parts.slice(keep))}</details>`;
}
function empty(title, text, action="", label="", symbol="leaf") {
  return `<div class="empty"><div class="empty-symbol">${icon(symbol)}</div><h3>${esc(title)}</h3><p>${esc(text)}</p>${action?button(action,label,"primary"):""}</div>`;
}
function panel(title, content, trailing="", cls="") {
  return `<section class="panel ${cls}"><div class="panel-header"><h2>${title}</h2>${trailing}</div>${content}</section>`;
}
function heading(title, subtitle, eyebrow="RESEARCH WORKSPACE", actions="") {
  return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${esc(title)}</h1><p class="subtitle">${esc(subtitle)}</p></div>${actions?`<div class="heading-actions">${actions}</div>`:""}</div>`;
}
function shell() {
  const project=state.detail?.config;
  $("#app").innerHTML=`<div class="shell"><aside class="sidebar" id="sidebar"><div class="brand-row"><button class="brand" data-do="home" aria-label="ScholarLoop home"><img src="/favicon.svg" alt=""><span>Scholar<span>Loop</span></span></button></div>
    <button class="btn primary new-button" data-do="new">${icon("plus")} New research</button>
    <div class="nav-label">WORKSPACE</div><nav aria-label="Research workflow">${tabs.map(([id,label,ico],i)=>`<button class="nav-item ${state.tab===id&&(state.run||id!=="overview")?"active":""}" data-tab="${id}">${icon(ico)}<span>${label}</span><span class="number">0${i+1}</span></button>`).join("")}</nav>
    <div class="project-list"><div class="nav-label">YOUR RESEARCH</div>${state.boot.runs.slice(0,12).map(r=>`<button class="nav-item project ${r.id===state.run?"active":""}" data-run="${esc(r.id)}"><span class="dot"></span><span class="title">${esc(r.title)}</span></button>`).join("") || '<p class="muted">&nbsp; No research projects yet</p>'}
    <button class="nav-item project" data-do="import">${icon("folder")} Open existing research</button><button class="nav-item project" data-do="sample">${icon("book")} Explore a real run</button></div>
    <div class="sidebar-bottom"><button class="nav-item" data-do="connection">${icon("sliders")} Connections</button><button class="nav-item ${!state.run&&state.tab==="overview"?"active":""}" data-do="home">${icon("grid")} All research</button><div class="local-status"><span class="status-dot"></span> LOCAL WORKSPACE <span>·</span> Local</div></div></aside>
    <div class="workspace"><header class="topbar"><div class="breadcrumb"><button class="breadcrumb-home" data-do="home" title="View all research">Workspace</button><span>/</span>${project?`<button class="breadcrumb-project" data-tab="overview" title="Research overview">${esc(project.title)}</button>`:`<strong>${state.tab==="overview"?"All research":esc(tabs.find(t=>t[0]===state.tab)?.[1]||"All research")}</strong>`}</div><div class="top-actions"><button class="connection ${state.boot.connected?"":"off"}" data-do="connection"><span class="status-dot"></span>${state.boot.connected?"API key configured":"API key not set"}</button><div class="avatar" title="Local researcher">SL</div></div></header><main id="main" class="content" tabindex="-1"></main></div></div>`;
  renderMain();
}
function nextAction() {
  const d=state.detail, s=d.session, r=d.review;
  if(d.config.readonly) return ["files","Explore archived research","Browse the original experiments and reviews in this read-only historical sample.","Browse artifacts",true];
  if(busy()) return ["experiments","Research is in progress","Background tasks save artifacts and logs while you explore existing results.","View progress",true];
  if(d.config.mode==="demo") return ["demo","Try the full experiment funnel","Explore the workflow with a scripted model and simulated scores. No API key or GPU required.","Run offline demo"];
  if(r.release_error) return ["review","The evidence has changed and needs a new review",r.release_error,"Review again"];
  if(r.status==="NEEDS_EXPERIMENTS") return ["agenda_plan","The review identified missing evidence","Plan experiments for the unresolved questions, inspect the protocol, then run it.","Plan follow-up experiments"];
  if(r.status==="READY_FOR_AUTHOR") return ["approve","The argument passed review and awaits your approval","Read the scientific review first. Your approval applies only to the current evidence version.","Review and approve"];
  if(r.status==="READY_TO_WRITE") return d.writing.stage==="reviewed" ? ["files","Your paper and reviews are ready","Read the paper, reviewer feedback, and the experimental evidence behind each claim.","View artifacts",true] : ["write","Write from reviewed evidence","Approve the outline and key points, then generate the manuscript and submit it for review.","Continue writing"];
  if((!s.stage||s.stage==="new")&&!d.entries.length) return ["literature","Turn curiosity into a research question","Describe your interests, hypotheses, and boundaries to start a grounded literature review.","Start literature review"];
  if(s.stage==="reviewed"&&!s.direction_map?.length) return ["directions","Find directions worth exploring","Read the literature review, then generate directions that support, challenge, or extend your hypotheses.","Generate directions"];
  if(s.stage==="reviewed") return ["ideas","Choose where the research goes next","Choose the directions that matter to you, or add your own questions, then generate concrete ideas.","Choose directions and generate ideas"];
  if(s.stage==="ideated") return ["select","Which idea is worth testing?","Compare hypotheses, experiment plans, and rankings. You make the final selection.","Select a research idea"];
  if(!d.entries.length) return ["campaign","Put your research idea to the test","Run controls and treatments through the evaluation funnel, recording every measurement and failure.","Start experiments"];
  return ["review","Review what the recorded results support","Four scientific reviewers examine each claim, the experiment design, and alternative explanations.","Start scientific review"];
}
function callout() {
  const [action,title,text,label,navigate]=nextAction();
  const btn=navigate?`<button class="btn primary" data-tab="${action}">${label} ${icon("arrow")}</button>`:button(action,label,"primary",action==="approve"?{review_hash:state.detail.review.hash}:{});
  const d=state.detail, finished=d.review.status==="READY_TO_WRITE"&&d.writing.stage==="reviewed", ready=finished||d.config.readonly;
  return `<div class="callout${ready?" ready":""}"><div class="callout-icon">${icon(busy()?"clock":finished?"check":ready?"folder":"bulb")}</div><div><h3>${esc(title)}</h3><p>${esc(text)}</p></div>${btn}</div>`;
}
function progress() {
  const d=state.detail,s=d.session;
  if(d.config.mode==="demo")return `<div class="workflow">${["smoke","verify","full"].map((tier,i)=>`<button class="workflow-step ${d.entries.some(e=>e.fidelity?.includes(tier))?"done":""}" data-tab="experiments"><span class="step-circle">${i+1}</span><strong>${tier}</strong><small>${[1,3,5][i]} random seeds</small></button>`).join("")}</div>`;
  let stage=!s.stage||s.stage==="new"?0:s.stage==="reviewed"?1:s.stage==="ideated"?1:2;
  if(d.entries.length)stage=3;if(d.review.status==="READY_TO_WRITE")stage=4;
  if(d.writing.stage==="reviewed")stage=5;
  return `<div class="workflow">${[["research","Literature review","Ground the question"],["ideas","Research Ideas","Choose the direction"],["experiments","Run experiments","Record the results"],["review","Scientific Review","Approve the argument"],["writing","Writing","Write from evidence"]].map(([tab,label,note],i)=>`<button class="workflow-step ${i<stage?"done":i===stage?"current":""}" data-tab="${tab}"><span class="step-circle">${i<stage?icon("check"):"0"+(i+1)}</span><strong>${label}</strong><small>${note}</small></button>`).join("")}</div>`;
}
function stats() {
  const d=state.detail, p=profile(), full=d.entries.filter(e=>e.score!==null&&e.score!==undefined&&e.fidelity?.includes("full"));
  const best=full.length?(p?.direction==="maximize"?Math.max:Math.min)(...full.map(e=>e.score)):null;
  const values=[["Experiment records",d.entries.length,"Kept, discarded, and failed runs","flask"],
    ["Best full evaluation",best===null?"—":Number(best).toFixed(3),p?.metric || "Awaiting measurements","grid"],
    ["Open review issues",d.review.issues.filter(i=>i.status==="open").length,"Questions still to resolve","shield"],
    ["Model cost",d.config.readonly?"—":`$${Number(d.spend.usd||0).toFixed(2)}`,d.config.readonly?"Recorded in this run's README":d.config.mode==="demo"?"No API calls · offline demo":`UI estimate · cap $${d.config.spend_cap}`,"leaf"]];
  return `<div class="stats">${values.map(([label,value,note,ico])=>`<div class="stat"><div class="stat-label">${label}${icon(ico)}</div><div class="stat-value">${esc(value)}</div><div class="stat-note">${esc(note)}</div></div>`).join("")}</div>`;
}
// An arm's own name, without the protocol prefix every arm of a study shares.
const shortId=id=>String(id??"").replace(/^protocol_[0-9a-f]{8,}_/,"");
// One bar per fully evaluated run, from zero: the runs are separate arms, not a series over time.
function chart() {
  const entries=state.detail.entries.filter(e=>e.score!==null&&e.score!==undefined&&e.fidelity?.includes("full"));
  if(!entries.length)return empty("Awaiting full evaluation results","After the full stage completes, compare results at the same evaluation tier here.","","","flask");
  const vals=entries.map(e=>Number(e.score)), top=Math.max(...vals,0)||1, row=30, left=190, span=320;
  const clip=text=>text.length>28?text.slice(0,27)+"…":text;
  return `<div class="chart"><svg viewBox="0 0 570 ${entries.length*row+6}" role="img" aria-label="Full evaluation result of each run">${entries.map((e,i)=>{const w=Math.max(2,Math.max(0,vals[i])/top*span), y=i*row+6;
    return `<g><title>${esc(e.id)}: ${vals[i]}</title><text class="chart-label chart-name" x="${left-10}" y="${y+14}" text-anchor="end">${esc(clip(shortId(e.id)))}</text><rect class="chart-track" x="${left}" y="${y}" width="${span}" height="18" rx="3"/><rect class="chart-bar" x="${left}" y="${y}" width="${w.toFixed(1)}" height="18" rx="3"/><text class="chart-label chart-value" x="${left+span+10}" y="${y+14}">${esc(vals[i])}</text></g>`;}).join("")}</svg></div><p class="chart-caption">${esc(profile()?.metric||"Score")} at the full tier, one bar per run · Check seeds and conditions before comparing</p>`;
}
function recentArtifacts() {
  const preferred=["paper.md","scientific_review.md","ideas.md","directions.md","lit_review.md","demo.md","experiments.jsonl"];
  const files=preferred.filter(fileExists).slice(0,4);
  return files.length?files.map(f=>`<div class="list-row"><div class="file-icon">${icon("file")}</div><div class="grow"><strong>${esc(f)}</strong><small>${{ "paper.md":"Manuscript","scientific_review.md":"Scientific review and issues","ideas.md":"Candidate ideas and rankings","directions.md":"Research direction map","lit_review.md":"Literature review","demo.md":"Offline demo notes","experiments.jsonl":"Raw experiment records"}[f]}</small></div>${artifactButton(f,"Open","ghost small")}</div>`).join(""):empty("Your artifacts will appear here","Every step of research, experimentation, and writing leaves a record you can inspect.","","","folder");
}
function jobPanel() {
  const jobs=state.detail.jobs, selected=jobs.find(j=>j.id===state.job);
  // A recorded run that never ran anything here has no task history to show.
  if(state.detail.config.readonly&&!jobs.length)return "";
  const activity=state.detail.activity||[];
  const events=activity.length?`<details class="agent-activity"><summary>Recent model activity · ${activity.length} events</summary>${activity.map(a=>`<div class="list-row"><div class="grow"><strong>${esc(a.agent)}</strong><small>Attempt ${esc(a.attempt)} · ${timeText(a.timestamp)}</small></div>${tag(a.event==="started"?"Started":({accepted:"Validated",rejected:"Validation failed",invalid_json:"Invalid output format",exception:"Request error"}[a.status]||a.status),a.status==="accepted"?"green":a.event==="started"?"blue":"amber")}</div>`).join("")}</details>`:"";
  return panel("Task history",(jobs.length?`<div>${jobs.slice(0,6).map(j=>`<button class="list-row task-row ${j.id===state.job?"selected":""}" data-job="${j.id}">${icon(j.status==="running"?"clock":"file")}<div class="grow"><strong>${esc(j.label)}</strong><small>${timeText(j.started)}</small></div>${tag(statusLabels[j.status],statusColors[j.status])}</button>`).join("")}</div>${selected?`<div class="task-summary row space-between"><span>${esc(selected.label)} · ${statusLabels[selected.status]}</span>${selected.status==="running"?`<button class="btn small danger" data-cancel="${selected.id}">${icon("stop")} Stop task</button>`:""}</div><pre class="log" id="job-log">${esc(state.log)||"Waiting for task output…"}</pre>`:""}`:empty("No tasks yet","Start a research or experiment task to see live logs here.","","","clock"))+events);
}
function overview() {
  const d=state.detail;
  return heading(d.config.title,"A record of every step, with you in control of the key decisions.","RESEARCH OVERVIEW",`<button class="btn" data-do="settings" ${d.config.readonly||busy()?`disabled title="${d.config.readonly?"This archive is read-only.":"Wait for the current task to finish."}"`:""}>${icon("sliders")} Research settings</button><a class="btn" href="/api/runs/${encodeURIComponent(state.run)}/export">${icon("download")} Export records</a><button class="btn danger" data-delete="${esc(state.run)}" ${busy()?'disabled title="Stop the running task before deleting this research."':""}>Delete research</button>`)+callout()+stats()+panel("Research progress",progress(),tag(d.config.mode==="demo"?"Offline demo":d.config.readonly?"Archive":"Human-guided research","outline"))+`<div class="spacer"></div><div class="grid-two"><div class="stack">${panel("Experiment results",chart(),`<button class="btn ghost small" data-tab="experiments">View experiments ${icon("arrow")}</button>`)}${jobPanel()}</div><div class="stack">${panel("Recent artifacts",recentArtifacts(),`<button class="btn ghost small" data-tab="files">View all ${icon("arrow")}</button>`)}${panel("Configuration",`<div class="panel-body"><div class="setting-line"><span>Experiment domain</span><strong>${esc(profile()?.name||d.config.profile)}</strong></div><div class="setting-line"><span>Model</span><small>${esc(d.config.model)}</small></div><div class="setting-line"><span>Request limit per stage</span><strong>${d.config.max_calls}</strong></div><div class="setting-line"><span>Research budget cap</span><strong>$${d.config.spend_cap}</strong></div><p class="form-note">Costs are estimated from tokens. An in-flight request may take the total slightly over the cap.</p></div>`)}</div></div>`;
}
function research() {
  const s=state.detail.session, dirs=s.direction_map||[];
  return heading("Find the next question in the literature","Build a foundation, then choose directions that support, challenge, or extend your hypotheses.","01 / LITERATURE & DIRECTIONS")+
    `<div class="action-row">${button("literature",fileExists("lit_review.md")?"Revise literature review":"Start literature review","primary")}${button("directions",dirs.length?"Revise directions":"Generate directions")}</div>`+
    (s.steer?.interest?panel("Your research intent",`<div class="panel-body"><p>${esc(s.steer.interest)}</p>${s.steer.hunches?.length?`<div class="tags">${s.steer.hunches.map(h=>tag(h)).join("")}</div>`:""}</div>`)+`<div class="spacer"></div>`:"")+
    panel("Literature review",fileExists("lit_review.md")?`<div class="list-row"><div class="file-icon">${icon("book")}</div><div class="grow"><strong>Literature review and research gaps</strong><small>Themes, paper notes, open questions, and sources</small></div>${artifactButton("lit_review.md","Read review")}</div>`:empty("Start with your question","Share your interests and hypotheses to focus the literature search on questions that matter to you."))+
    `<div class="spacer"></div><div class="section-title"><h2>Research directions ${dirs.length?tag(dirs.length+" directions"):""}</h2>${dirs.length?button("ideas","Choose directions and generate ideas","soft"):""}</div>`+
    (dirs.length?`<div class="cards idea-list">${dirs.map(directionCard).join("")}</div>`:panel("Direction map",(state.detail.config.readonly?recorded("directions.md","The direction map is kept as a document","Read directions"):empty("Turn the literature into directions","The direction map identifies testable questions, low-cost first experiments, and likely failure modes."))))+
    `<div class="spacer"></div>${jobPanel()}`;
}
function ideas() {
  const ideas=state.detail.session.ideas||[];
  return heading("Choose an idea worth testing","Compare hypotheses, feasibility, and experiment plans. Rankings inform your choice; you set the direction.","02 / RESEARCH IDEAS")+
    `<div class="action-row">${button("ideas","Generate candidate ideas","primary")}${button("refine_ideas","Give feedback and refine")}${button("inject","Add your own idea")}${button("decide","Manage idea decisions")}${button("pilot_plan","Plan a pilot")}</div>`+
    (ideas.length?`<div class="cards idea-list">${ideas.map(ideaCard).join("")}</div>`:panel("Candidate ideas",(state.detail.config.readonly?recorded("ideas.md","The ideas and their tournament are kept as a document","Read ideas"):empty("Develop a testable hypothesis","Complete the literature review and choose directions, or submit your own idea with supporting sources."))))+
    `<div class="spacer"></div>${protocols("pilot")}<div class="spacer"></div>${jobPanel()}`;
}
function experiments() {
  const d=state.detail;
  return heading("Put ideas to the test","From small-scale screening to full evaluation, every run is recorded.","03 / EXPERIMENTS")+
    `<div class="action-row">${button(d.config.mode==="demo"?"demo":"campaign",busy()?"Experiments running":d.config.mode==="demo"?"Run offline demo":"Start / resume experiments","primary")}${button("iterate","Assess next research steps")}${button("review","Submit for scientific review")}</div>`+
    stats()+panel("Full evaluation results",chart())+`<div class="spacer"></div>`+
    panel("Experiment records",d.entries.length?`<div class="table-wrap"><table><thead><tr><th>Experiment ID</th><th>Stage</th><th>Role / purpose</th><th>${esc(profile()?.metric||"Metric")}</th><th>Result</th><th>Hypothesis</th></tr></thead><tbody>${d.entries.slice().reverse().map(e=>`<tr><td class="mono" title="${esc(e.id)}">${esc(shortId(e.id))}</td><td>${tag(e.fidelity?.join(" / ")||"—","outline")}</td><td>${esc(e.role||e.purpose||"exploration")}</td><td class="score">${e.score===null||e.score===undefined?"—":esc(e.score)}</td><td>${tag({kept:"Kept",discarded:"Discarded",killed:"Failed",pending:"Pending"}[e.verdict]||e.verdict,e.verdict==="kept"?"green":e.verdict==="killed"?"red":"")}</td><td class="long" title="${esc(e.hypothesis?.claim)}">${esc(e.hypothesis?.claim)}</td></tr>`).join("")}</tbody></table></div>`:empty("No experiments yet","Select a research idea, then set the rounds, parallel workers, and cumulative training budget."),fileExists("experiments.jsonl")?artifactButton("experiments.jsonl","Raw records"):"")+
    `<div class="spacer"></div>${fileExists("research_iterations.json")?panel("Research iteration recommendations",`<div class="list-row"><div class="grow"><strong>Competing explanations, distinguishing experiments, and remaining uncertainty</strong><small>Recommendations need scientific review before becoming a frozen experiment protocol.</small></div>${artifactButton("research_iterations.json","View recommendations")}</div>`)+`<div class="spacer"></div>`:""}${jobPanel()}`;
}
function protocols(kind="agenda") {
  const ps=state.detail.protocols.filter(p=>p.kind===kind);
  if(!ps.length)return "";
  return panel(kind==="pilot"?"Idea pilot protocols":"Follow-up experiment protocols",ps.map(p=>`<div class="list-row"><div class="file-icon">${icon("flask")}</div><div class="grow"><strong class="mono">${esc(p.id.slice(0,16))}…</strong><small>${esc(p.status)}</small></div>${artifactButton(p.path,"View protocol")}${button(kind+"_run","Run / resume","small",{protocol_hash:p.id})}</div>`).join(""));
}
const reviewRoles={implementation:["Implementation","Does the code do what the argument says it does?"],experimental:["Experiment design","Are the comparisons fair and the numbers right?"],
  falsification:["Falsifiability","Could each claim have failed, and were the controls run?"],argument:["Argument and evidence","Does the recorded evidence support every statement?"]};
const words=value=>String(value??"").replaceAll("_"," ");
// The verdict leads the page: what was decided, and the counts behind it.
function reviewVerdict(r) {
  const tone=r.release_error?"stale":r.status==="READY_TO_WRITE"?"approved":"waiting";
  const checks=r.reports.flatMap(report=>report.claim_checks||[]), supported=checks.filter(c=>String(c.status).startsWith("supported")).length;
  const open=r.issues.filter(i=>i.status==="open").length, passed=r.reports.filter(report=>report.verdict==="pass").length;
  const stat=(value,label,alert=false)=>`<div class="review-stat${alert?" alert":""}"><b>${value}</b><span>${label}</span></div>`;
  return `<section class="panel review-verdict ${tone}"><div class="review-verdict-head"><div class="review-mark">${icon(tone==="approved"?"check":"clock")}</div><div class="grow"><div class="idea-kicker">Verdict</div><h2>${esc(reviewLabels[r.status]||r.status)}</h2><p class="form-note">Approval applies only to the report you reviewed. New experiments or changes to the argument or code require a new review.</p></div>${tag(r.release_error?"New review required":r.status,r.release_error?"red":tone==="approved"?"green":"amber")}</div>`+
    `<div class="review-stats">${stat(`${passed}<small>/${r.reports.length}</small>`,"Reviewers passed",passed<r.reports.length)}${stat(`${supported}<small>/${checks.length}</small>`,"Claim checks supported",supported<checks.length)}${stat(r.issues.length,"Issues raised across rounds")}${stat(open,"Issues still open",open>0)}</div>`+
    `<div class="review-verdict-foot">${fileExists("scientific_review.md")?artifactButton("scientific_review.md","Read full review"):""}${r.hash?`<span class="mono">Version ${esc(r.hash.slice(0,24))}…</span>`:""}</div></section>`;
}
function reviewerCard(report, issues) {
  const [name,asks]=reviewRoles[report.role]||[report.role,""], pass=report.verdict==="pass";
  const checks=report.claim_checks||[], supported=checks.filter(c=>String(c.status).startsWith("supported")).length;
  const mine=issues.filter(i=>i.role===report.role), open=mine.filter(i=>i.status==="open").length;
  return `<article class="panel reviewer-card ${pass?"pass":"flag"}"><header><div class="grow"><h3>${esc(name)}</h3><p class="reviewer-asks">${esc(asks)}</p></div>${tag(report.verdict,pass?"green":"amber")}</header>`+
    `<div class="reviewer-facts"><span><b>${supported}/${checks.length}</b> claims supported</span><span><b>${mine.length}</b> issues raised</span><span${open?' class="alert"':""}><b>${open}</b> still open</span></div>${leadPoints(report.summary)}</article>`;
}
// Open issues first and unfolded; resolved ones by severity, each a single line until opened.
function reviewIssues(issues) {
  const rest=[...issues], take=test=>{const mine=rest.filter(test);mine.forEach(i=>rest.splice(rest.indexOf(i),1));return mine;};
  const groups=[["Open","Still to resolve before approval.",take(i=>i.status==="open"),true],["Major, closed","Resolved, verified or ruled out of scope.",take(i=>i.severity==="major"||i.severity==="blocking"),false],
    ["Minor, closed","Noted or resolved; none of them blocks writing.",take(()=>true),false]];
  const part=(label,text)=>text?`<div class="idea-label">${label}</div>${points(text)}`:"";
  return groups.filter(g=>g[2].length).map(([title,note,mine,open])=>`<details class="file-group issue-group"${open?" open":""}><summary><span class="file-group-title">${title}</span><span class="file-group-count">${mine.length}</span><span class="file-group-note">${note}</span></summary>${mine.map(i=>
    `<details class="review-issue ${i.status==="open"?"open-issue":""}"><summary><span class="issue-severity ${esc(i.severity||"")}">${esc(i.severity||"issue")}</span><span class="issue-title">${esc(i.title||i.target||i.id)}</span><small>${esc((reviewRoles[i.role]||[i.role])[0])} · ${esc(words(i.action))}</small>${tag(words(i.status),i.status==="open"?"amber":"green")}</summary><div class="issue-body">${part("Observation",i.observation||i.description)}${part("Required action",i.required_action)}${part("Resolution criterion",i.resolution_criterion)}${part("How it was closed",i.resolution?.reason)}</div></details>`).join("")}</details>`).join("");
}
function review() {
  const r=state.detail.review, ready=r.status==="READY_FOR_AUTHOR"&&!r.release_error;
  return heading("Review the argument before writing","Four reviewers examine the evidence. Author approval is tied to the current research version.","04 / SCIENTIFIC REVIEW")+
    (r.release_error?`<div class="notice error">The current approval is no longer valid: ${esc(r.release_error)}</div>`:"")+
    `<div class="action-row">${button("review",r.status?"Give feedback and review again":"Start scientific review","primary")}${ready?button("approve","Approve this version for writing","soft",{review_hash:r.hash}):""}${r.status?button("review_status","Validate review status"):""}${r.status==="NEEDS_EXPERIMENTS"?button("agenda_plan","Plan follow-up experiments","soft"):""}</div>`+
    (r.status?reviewVerdict(r):panel("Scientific Review",state.detail.config.readonly?recorded("review.md","This run predates the scientific review; its reviewer panel report is kept as a document","Read panel report"):empty("Establish what the evidence supports","After experiments, examine the implementation, experiment design, falsifiability, and final argument.")))+
    (r.reports.length?`<div class="spacer"></div><div class="section-title"><h2>The four reviewers</h2></div><div class="tool-grid">${r.reports.map(report=>reviewerCard(report,r.issues)).join("")}</div>`:"")+
    (r.issues.length?`<div class="spacer"></div>`+panel(`Review issues <span class="file-group-count">${r.issues.length}</span>`,reviewIssues(r.issues)):"")+
    `<div class="spacer"></div>${protocols()}<div class="spacer"></div>${jobPanel()}`;
}
// The manuscript leads; the four stages below show how it was built and what each one left behind.
const writeSteps=[["outline.md","Outline","The narrative, and the question each section answers."],["key_points.md","Key points","What each section will say, and the evidence it draws on."],
  ["paper.md","Manuscript","Sections drafted from the approved key points."],["review.md","Panel review","Simulated reviewers read the finished draft."]];
function writing() {
  const d=state.detail, stages=["new","outlined","keypointed","drafted","reviewed"], current=Math.max(0,stages.indexOf(d.writing.stage));
  const labels=["Generate outline","Approve outline and generate key points","Approve key points and draft manuscript","Submit manuscript for review","Writing complete"];
  const m=d.writing.summary||{}, done=current===4;
  const stat=(value,label)=>`<div class="review-stat"><b>${value}</b><span>${label}</span></div>`;
  const hero=current||m.title?`<section class="panel manuscript-hero${done?" done":""}"><div class="manuscript-head"><div class="idea-kicker">${done?"Manuscript · writing complete":`Manuscript · stage ${Math.min(current+1,4)} of 4`}</div><h2>${esc(m.title||"Untitled manuscript")}</h2></div>`+
    `<div class="review-stats three">${stat(m.sections||"—","Sections drafted")}${stat(m.words?Number(m.words).toLocaleString("en-US"):"—","Words")}${stat(m.reviewers||"—","Simulated reviewers' reports")}</div>`+
    `<div class="manuscript-foot">${fileExists("paper.md")?artifactButton("paper.md","Read the manuscript","primary"):""}${fileExists("paper_traced.md")?artifactButton("paper_traced.md","Sentence-by-sentence trace"):""}${fileExists("evidence_map.md")?artifactButton("evidence_map.md","Evidence map"):""}<span class="manuscript-note">${icon("shield")} Every sentence cites an approved statement.</span></div></section>`:"";
  return heading("Turn tested ideas into a paper","Build an evidence-backed paper through an outline, key points, manuscript, and review.","05 / WRITING")+
    ((d.review.status!=="READY_TO_WRITE"||d.review.release_error)&&!d.config.readonly?`<div class="notice">Writing requires a successful scientific review and author approval. <button class="btn ghost small" data-tab="review">Go to scientific review ${icon("arrow")}</button></div>`:"")+
    hero+`<div class="section-title write-title"><h2>How it was written</h2><small>${done?"All four stages complete":`Next: ${esc(labels[current])}`}</small></div>`+
    `<div class="write-track">${writeSteps.map(([file,title,desc],i)=>{const state_=i<current?"done":i===current?"current":"todo";
      return `<article class="panel write-step ${state_}"><div class="write-step-top"><div class="write-circle">${state_==="done"?icon("check"):i+1}</div><div class="write-line"></div></div><div class="write-stage-label">Stage 0${i+1}</div><h3>${title}</h3><p>${desc}</p><div class="write-step-foot">${fileExists(file)?artifactButton(file,"Read","ghost small"):tag(state_==="current"?"Up next":"Not yet generated",state_==="current"?"amber":"outline")}</div></article>`;}).join("")}</div>`+
    `<div class="action-row">${current<4?button("write",labels[current],"primary"):""}${button("write_feedback","Give feedback and revise")}${current<4?button("write_auto","Complete remaining writing stages"):""}</div>`+jobPanel();
}
const toolNotes={evaluate:"Inspect experiment records, review validity, and model logs without changing them.",traces:"Summarize accepted, rejected, failed, and incomplete model outputs.",inject_faults:"Inject errors into artifact copies to check whether validators detect them.",baseline:"Use a live model for a single writing pass, then apply the same checks.",decision_bench:"Evaluate synthetic research decisions without a model API.",decision_policy:"Compare model strategies with request limits for each task and strategy.",suite:"Run random search evaluations across the tasks in a manifest.",compare:"Compare random search and the agent under the same budget rules."};
// Tools in two shelves: what is free to run, and what spends model budget.
const toolIcons={evaluate:"shield",traces:"file",inject_faults:"flask",decision_bench:"sliders",suite:"grid",baseline:"pen",decision_policy:"bulb",compare:"play"};
const toolShelves=[["Offline checks","Run on the records already on disk. No model calls, no cost.",["evaluate","traces","inject_faults","decision_bench","suite"],"free","No model calls"],
  ["Live comparisons","Call the model to produce something to compare against. These spend from the research budget.",["baseline","decision_policy","compare"],"live","Uses the model"]];
function toolsPage() {
  const actions=state.boot.actions.filter(a=>a.group==="tools"), placed=new Set(toolShelves.flatMap(shelf=>shelf[2]));
  const shelves=toolShelves.map(([title,note,ids,kind,badge])=>[title,note,ids.map(id=>actions.find(a=>a.id===id)).filter(Boolean),kind,badge]);
  shelves[1][2].push(...actions.filter(a=>!placed.has(a.id)));
  return heading("Evaluate research with consistent standards","Run the existing diagnostics, reliability checks, and comparative evaluations.","EVALUATION & TOOLS")+
    shelves.filter(shelf=>shelf[2].length).map(([title,note,items,kind,badge])=>`<section class="tool-shelf ${kind}"><div class="tool-shelf-head"><h2>${title}</h2><span class="file-group-count">${items.length}</span><p>${note}</p></div><div class="tool-cards">${items.map(a=>
      `<article class="panel tool-tile"><div class="tool-tile-icon">${icon(toolIcons[a.id]||"sliders")}</div><div class="grow"><h3>${esc(a.label)}</h3><p>${esc(toolNotes[a.id]||"")}</p></div><div class="tool-tile-foot"><span class="tool-cost">${a.id==="compare"?"Random search alone is free":badge}</span>${button(a.id,"Configure and run")}</div></article>`).join("")}</div></section>`).join("")+
    `<div class="spacer"></div>${jobPanel()}`;
}
function filesPage() {
  const files=state.detail.artifacts.filter(f=>f.path.toLowerCase().includes(state.filter.toLowerCase()));
  return heading("A record of the entire research process","Read, inspect, and export files from your local research directory.","RESEARCH ARTIFACTS",`<a class="btn" href="/api/runs/${encodeURIComponent(state.run)}/export">${icon("download")} Export research records</a>`)+
    `<input class="file-search" id="file-search" placeholder="Search filenames, e.g. paper, review, protocol…" aria-label="Search artifacts" value="${esc(state.filter)}">`+
    panel(`<span id="file-count" aria-live="polite">${fileCount(files)}</span>`,`<div class="file-list" id="file-list">${fileRows(files)}</div>`)+`<p class="subtitle mono">${esc(state.detail.config.path)}</p>`;
}
// Artifacts by the stage that produced them, in the order a reader would look for them.
const fileGroups=[
  ["Paper and writing","The manuscript, its traced edition, the evidence map and the reviewers' reports.",
    f=>/^(manuscripts\/|paper\d*\/)/.test(f)||/^(paper|paper_traced|evidence_map|review|outline|key_points)\.md$/.test(f)||/^(writing|approved_evidence|draft_checkpoint)\.json$/.test(f)],
  ["Scientific review","The argument, the critics' reports and the author's release.",
    f=>/^(scientific_review|argument_|review_history\/|experiment_requests|pending_decisions|research_iterations)/.test(f)],
  ["Experiments","The ledger of every run, frozen protocols and the code version of each arm.",
    f=>/^(experiments\.jsonl|registry\/|experiment_agenda\/|run\.md|process\.|calibration|handoff\.json|usage\.json)/.test(f)],
  ["Literature and ideas","The steer, the literature review, the direction map, the ideas and their tournament.",
    f=>/^(steer|lit_review|directions|ideas|idea_|tournament|session)/.test(f)],
  ["Reliability checks","Fault injection, trace statistics and the claim audit for this run.",f=>/^reliability\//.test(f)],
  ["Other","Notes, logs and everything else.",()=>true],
];
function fileRow(f) {
  return `<div class="list-row"><div class="file-icon">${icon("file")}</div><div class="grow"><button class="file-name" data-file="${esc(f.path)}">${esc(f.path)}</button><small>${f.size<1024?f.size+" B":(f.size/1024).toFixed(1)+" KB"}</small></div>${artifactButton(f.path,"Read","ghost small")}</div>`;
}
function fileRows(files) {
  if(!files.length)return empty("No matching files","Research artifacts will appear here as they are generated.","","","folder");
  const rest=[...files];
  let shown=0;
  return fileGroups.map(([title,note,belongs])=>{
    const mine=rest.filter(f=>belongs(f.path));
    mine.forEach(f=>rest.splice(rest.indexOf(f),1));
    if(!mine.length)return "";
    // Only the first group starts unfolded, so every section title is in view; a search unfolds all.
    const open=state.filter||!shown++;
    return `<details class="file-group"${open?" open":""}><summary><span class="file-group-title">${esc(title)}</span><span class="file-group-count">${mine.length}</span><span class="file-group-note">${esc(note)}</span></summary>${mine.map(fileRow).join("")}</details>`;
  }).join("");
}
function fileCount(files) {
  return `${files.length} of ${state.detail.artifacts.length} research files`;
}
function updateFileSearch() {
  const files=state.detail.artifacts.filter(f=>f.path.toLowerCase().includes(state.filter.toLowerCase()));
  $("#file-list").innerHTML=fileRows(files);
  $("#file-count").textContent=fileCount(files);
}
function researchCards(tab="overview") {
  const destination=tabs.find(t=>t[0]===tab)?.[1]||"Overview";
  return `<div class="cards">${state.boot.runs.map(r=>`<article class="project-card"><button class="project-open" data-run="${esc(r.id)}" data-target-tab="${esc(tab)}" aria-label="Open ${esc(r.title)}">${tag(r.readonly?"Archive":r.mode==="demo"?"Offline demo":"Research project",r.mode==="demo"?"amber":"green")}<h3>${esc(r.title)}</h3><p class="muted">${esc(r.profile.replace(".yaml",""))} · ${timeText(r.created)}</p></button><div class="project-card-footer"><button class="btn ghost small project-destination" data-run="${esc(r.id)}" data-target-tab="${esc(tab)}">Open ${esc(destination)} ${icon("arrow")}</button><button type="button" class="btn ghost small danger" data-delete="${esc(r.id)}" aria-label="Delete ${esc(r.title)}">Delete</button></div></article>`).join("")}</div>`;
}
function researchPicker() {
  const label=tabs.find(t=>t[0]===state.tab)?.[1]||"Research";
  return heading(label,`Choose a research project to open its ${label.toLowerCase()} page.`,"SELECT RESEARCH",`<button class="btn primary" data-do="new">${icon("plus")} New research</button>`)+
    (state.boot.runs.length?researchCards(state.tab):`<section class="panel">${empty("No research projects yet","Create a research project or open an existing research directory.")}<div class="panel-body"><button class="btn" data-do="import">${icon("folder")} Open existing research</button></div></section>`);
}
function home() {
  return heading("Research starts with a good question.","Your entire research loop, in one place.","YOUR RESEARCH WORKSPACE")+
    `<section class="welcome"><div><div class="eyebrow">FROM CURIOSITY TO EVIDENCE</div><h1>Put ideas to the test.<br>Ground claims in evidence.</h1><p>From the first paper to the final review, ScholarLoop organizes your research. The model explores; you make the key decisions.</p><div class="welcome-actions"><button class="btn primary" data-do="new">${icon("plus")} Start a research project</button><button class="btn" data-do="sample">Explore a real run ${icon("arrow")}</button></div><div class="welcome-links"><button class="btn ghost" data-do="demo">Try the offline demo ${icon("arrow")}</button></div></div><svg class="welcome-art" viewBox="0 0 340 340" aria-hidden="true"><circle cx="170" cy="170" r="138" fill="none" class="orbit-dashed" stroke-dasharray="3 7"/><circle cx="170" cy="170" r="102" fill="none" class="orbit" stroke-width="1"/><path d="M170 68a102 102 0 0 1 99 127" fill="none" class="orbit-arrow" stroke-width="3"/><path d="m261 183 8 14 13-9" fill="none" class="orbit-arrow" stroke-width="3"/><circle cx="170" cy="170" r="59" class="orbit-core"/><path d="M181 134c-28-4-44 16-35 39m13 29c28 4 44-16 35-39" fill="none" class="orbit-mark" stroke-width="3"/><circle cx="170" cy="170" r="7" class="orbit-dot"/><rect x="136" y="43" width="68" height="47" rx="12" class="orbit-card"/><path d="M158 55v23m0-23 12 3 12-3v23l-12 3-12-3m12-20v23" fill="none" class="orbit-symbol" stroke-width="1.5"/><rect x="232" y="214" width="68" height="47" rx="12" class="orbit-card"/><path d="M259 225h14m-11 0v8l-10 17h28l-10-17v-8m-13 17h18" fill="none" class="orbit-symbol" stroke-width="1.5"/><rect x="40" y="214" width="68" height="47" rx="12" class="orbit-card"/><path d="M65 225h18v26H65zM69 232h10m-10 6h10m-10 6h7" fill="none" class="orbit-symbol" stroke-width="1.5"/><circle cx="69" cy="147" r="4" class="orbit-dot"/><circle cx="238" cy="93" r="4" class="orbit-dot"/><circle cx="170" cy="273" r="4" class="orbit-dot"/></svg></section>`+
    `<div class="feature-grid">${[["book","Ideas grounded in sources","Search the literature around your question, choose directions, and compare hypotheses."],["flask","Traceable experiments","Run real experiments with independent scoring, recorded results, failures, and code provenance."],["shield","Reviewed arguments","Review the evidence, approve the key claims, then build the paper step by step."]].map(([ico,title,text])=>`<article class="panel feature">${icon(ico)}<h3>${title}</h3><p>${text}</p></article>`).join("")}</div>`+
    (state.boot.runs.length?`<section class="project-grid"><div class="section-title"><h2>Your research</h2><button class="btn ghost small" data-do="import">Open existing research ${icon("arrow")}</button></div>${researchCards("overview")}</section>`:"");
}
function renderMain() {
  const main=$("#main");
  if(!state.detail){main.innerHTML=state.tab==="overview"?home():researchPicker();return;}
  const d=state.detail;
    const notice=d.config.mode==="demo"?`<div class="notice">Offline demo · Model responses and scores are simulated to demonstrate the workflow. They are not research evidence.</div>`:d.config.readonly?`<div class="notice">Archive · A read-only recorded run. Its approval was checked on the machine that ran it and is not re-checked here.</div>`:"";
  main.innerHTML=notice+(d.warnings.length?`<div class="notice error">${d.warnings.map(esc).join("<br>")}</div>`:"")+({overview,research,ideas,experiments,review,writing,files:filesPage,tools:toolsPage}[state.tab]||overview)()+`<div class="footer-note"><span>ScholarLoop · A research loop that shows its work.</span><span>Files saved locally · No cloud deployment needed</span></div>`;
}

async function loadRun(id, tab="overview") {
  const navigation=++state.navigation;
  const detail=await api(`/api/runs/${encodeURIComponent(id)}`);
  if(navigation!==state.navigation)return;
  state.run=id;state.detail=detail;state.tab=tab;state.job=null;state.log="";state.offset=0;state.filter="";
  localStorage.setItem("scholarloop.run",id);
  if(detail.jobs.length)state.job=detail.jobs[0].id;
  shell();window.scrollTo(0,0);await updateLog();
}
function goHome() {
  state.navigation++;
  state.run=null;state.detail=null;state.tab="overview";state.job=null;state.filter="";state.log="";state.offset=0;
  localStorage.removeItem("scholarloop.run");shell();window.scrollTo(0,0);
}
function selectTab(tab) {
  if(!tabs.some(t=>t[0]===tab))return;
  state.navigation++;state.tab=tab;shell();window.scrollTo(0,0);
}
async function reload() {
  state.boot=await api("/api/bootstrap");
  if(state.run)state.detail=await api(`/api/runs/${encodeURIComponent(state.run)}`);
  shell();
}
function fieldHtml(f, value=f.default) {
  const id="field-"+f.name, req=f.required?"required":"", attr=`id="${id}" name="${esc(f.name)}" ${req}`;
  if(f.type==="checkbox")return `<div class="field checkbox"><input type="checkbox" ${attr} ${value?"checked":""}><label for="${id}">${esc(f.label)}</label></div>`;
  let input;
  if(f.type==="textarea"||f.type==="json")input=`<textarea ${attr} class="${f.type==="json"?"code":""}" placeholder="${esc(f.placeholder||"")}">${esc(typeof value==="object"?JSON.stringify(value,null,2):value)}</textarea>`;
  else if(f.type==="select")input=`<select ${attr}>${f.options.map(o=>{const [v,l]=Array.isArray(o)?o:[o,o];return `<option value="${esc(v)}" ${v===value?"selected":""}>${esc(l)}</option>`;}).join("")}</select>`;
  else input=`<input ${attr} type="${f.type==="number"?"number":f.type==="password"?"password":"text"}" value="${esc(value)}" ${f.min!==undefined?`min="${f.min}" max="${f.max}" step="${f.step||1}"`:""} placeholder="${esc(f.placeholder||"")}" ${f.name==="review_hash"?"readonly":""} ${f.type==="password"?'autocomplete="off"':""}>`;
  return `<div class="field"><label for="${id}">${esc(f.label)} ${f.required?'<span class="field-required">*</span>':""}</label>${input}${f.help?`<small>${esc(f.help)}</small>`:""}</div>`;
}
function dialog(title, subtitle, fields, onSubmit, submit="Confirm", note="", presets={}) {
  const modal=$("#modal");
  modal.className="";
  modal.innerHTML=`<form id="dialog-form"><div class="dialog-header"><div><h2 id="modal-title">${esc(title)}</h2><p>${esc(subtitle)}</p></div><button type="button" class="btn ghost icon-only" data-close aria-label="Close">${icon("close")}</button></div><div class="dialog-body">${note}<div class="form-error" id="form-error" role="alert" hidden></div>${fields.map(f=>fieldHtml(f,presets[f.name]??f.default)).join("")}</div><div class="dialog-footer"><button type="button" class="btn" data-close>Cancel</button><button class="btn primary" type="submit">${esc(submit)} ${icon("arrow")}</button></div></form>`;
  $("#dialog-form").addEventListener("submit",async e=>{
    e.preventDefault();const form=e.currentTarget,submitButton=$('button[type="submit"]',form);submitButton.disabled=true;
    const values={};for(const f of fields){const input=form.elements.namedItem(f.name);values[f.name]=f.type==="checkbox"?input.checked:input.value;}
    try{await onSubmit(values);modal.close();}catch(error){const box=$("#form-error");box.textContent=error.message;box.hidden=false;box.scrollIntoView({block:"nearest"});}finally{submitButton.disabled=false;}
  });
  if(!modal.open)modal.showModal();
}
function showAction(action, presets={}) {
  const spec=state.boot.actions.find(a=>a.id===action);if(!spec)return;
  const reason=actionReason(action);if(reason){toast(reason,true);return;}
  const s=state.detail.session;
  if(action==="literature")presets={interest:s.steer?.interest||"",hunches:(s.steer?.hunches||[]).join("\n"),avoid:(s.steer?.avoid||[]).join("\n"),...presets};
  if(action==="iterate")presets={question:state.detail.config.title,...presets};
  if(action==="approve")presets={review_hash:state.detail.review.hash||"",...presets};
  let note="";
  if(action==="approve")note=`<div class="notice">This records your formal approval of the current evidence version. Read the review report first. ${fileExists("scientific_review.md")?artifactButton("scientific_review.md","Read report"):""}</div>`;
  if(action.endsWith("_run"))note='<p class="form-note">Run the frozen protocol you have reviewed. Interrupted test-set accesses remain recorded and cannot be bypassed by rerunning.</p>';
  if(action==="write_auto")note='<p class="form-note">With valid scientific review approval, automatically advance through the outline, key points, manuscript, and review. All existing checks still apply.</p>';
  if(action==="demo")note='<p class="form-note">Run smoke → verify → full using a scripted model and simulated engine, with no paid API calls.</p>';
  if(action==="ideas")note='<p class="form-note">Enter direction numbers or your own directions. Leave blank to reuse the selected directions.</p>';
  dialog(spec.label,state.detail.config.title,spec.fields,async values=>{
    const job=await api(`/api/runs/${encodeURIComponent(state.run)}/actions`,{action,params:values});
    state.job=job.id;state.offset=0;state.log="";state.detail=await api(`/api/runs/${encodeURIComponent(state.run)}`);
    state.tab=spec.group==="tools"?"tools":spec.group==="research"?"research":spec.group;
    shell();toast("Task started. Artifacts and logs will update as it runs.");
  },action==="approve"?"Approve this evidence version":"Run",note,presets);
}
function projectFields(edit=false, demo=false) {
  return [
    {name:"title",label:"Research title / topic",type:"text",default:demo?"My first offline experiment":"",required:true,placeholder:"e.g. How regularization affects small MLPs"},
    ...(!edit?[{name:"mode",label:"Research mode",type:"select",default:demo?"demo":"live",options:[["live","Live research · Uses model API"],["demo","Offline demo · No API or GPU required"]]},
    {name:"profile",label:"Experiment domain",type:"select",default:demo?"image-classification.yaml":"digits-mlp.yaml",options:state.boot.profiles.map(p=>[p.id,p.name+" · "+p.metric])}]:[]),
    {name:"model",label:"Model ID",type:"text",default:state.boot.environment.default_model,required:true},
    {name:"spend_cap",label:"Cumulative model budget cap (USD)",type:"number",default:10,min:.01,max:100000,step:.01},
    {name:"max_calls",label:"Cumulative request limit per stage (including retries)",type:"number",default:100,min:1,max:100000}
  ];
}
function newProject(demo=false) {
  dialog("Start a research project","Give your curiosity a clear direction.",projectFields(false,demo),async values=>{
    const r=await api("/api/runs",values);state.boot=await api("/api/bootstrap");await loadRun(r.id);toast("Research project created");
  },"Create research project",'<p class="form-note">The offline demo uses a separate simulated domain. Live research budgets accumulate across literature, experiments, review, and writing.</p>');
}
function settings() {
  if(state.detail.config.readonly||busy()){toast(state.detail.config.readonly?"This archive is read-only.":"Wait for the current task to finish.",true);return;}
  dialog("Research settings","Changing budgets preserves requests and costs already incurred.",projectFields(true),async values=>{
    await api(`/api/runs/${encodeURIComponent(state.run)}/settings`,values);await reload();toast("Research settings updated");
  },"Save settings","",state.detail.config);
}
function confirmDelete(id) {
  const record=state.boot.runs.find(r=>r.id===id);
  if(!record)return;
  if(state.run===id&&busy()){toast("Stop the running task before deleting this research.",true);return;}
  dialog("Delete research?",record.title,[],async()=>{
    await api(`/api/runs/${encodeURIComponent(id)}/delete`,{confirmed:true});
    state.navigation++;
    state.boot=await api("/api/bootstrap");
    if(state.run===id)goHome();else shell();
    toast("Research removed from the workspace. Local files were kept.");
  },"Confirm deletion",'<p class="form-note">This removes the research from this workspace. Its local files will be kept; you can reopen the directory later using Open existing research.</p>');
  $('#dialog-form button[type="submit"]').className="btn danger";
}
function connection() {
  const env=state.boot.environment;
  const note=`<p class="form-note">The key is kept only in server memory and is never saved to research files. You can also set ANTHROPIC_API_KEY before launch. Saving a key does not make an API request.</p><div class="panel panel-body"><div class="setting-line"><span>Python</span><strong>${esc(env.python)}</strong></div>${Object.entries(env.dependencies).map(([name,yes])=>`<div class="setting-line"><span>${esc(name)}</span>${tag(yes?"Installed":"Not installed",yes?"green":"amber")}</div>`).join("")}<details><summary>Install dependencies</summary><pre class="mono">pip install -e ".[dev,llm,engines]"
# AG News also requires .[nlp]</pre></details></div><div class="spacer"></div>`;
  dialog("Connections & environment",state.boot.connected?"An API key is configured. You can replace or clear it.":"Connect a model, or try experiments with the offline demo.",[{name:"api_key",label:"Anthropic API Key (leave blank and save to clear)",type:"password",default:""}],async values=>{
    const result=await api("/api/connection",values);state.boot.connected=result.connected;shell();toast(result.connected?"API key saved for this server session":"API key cleared for this server session");
  },"Save connection",note);
}
function importProject() {
  dialog("Open existing research","Open an existing CLI research directory with its original files.",[
    {name:"path",label:"Research directory",type:"text",default:"",required:true,placeholder:"runs/my-study or an absolute path inside the workspace"},
    {name:"profile",label:"Experiment domain",type:"select",default:"",options:[["","Detect from records"],...state.boot.profiles.map(p=>[p.id,p.name])]}
  ],async values=>{const r=await api("/api/import",values);state.boot=await api("/api/bootstrap");await loadRun(r.id);},"Open research");
}

// Small, deliberately conservative Markdown renderer: raw HTML is always escaped.
function inline(text) {
  let s=esc(text);
  s=s.replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>");
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(all,label,url)=>{
    if(/^https?:\/\//i.test(url))return `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    if(/^[\w./-]+\.(md|json|jsonl|txt|csv|yaml|yml|py)$/.test(url)) {
      const base=state.selectedFile.split("/").slice(0,-1).join("/");
      return `<button class="file-name" data-file="${esc((base?base+"/":"")+url)}">${label}</button>`;
    }
    return all;
  });
  return s;
}
function markdown(text) {
  const lines=text.split(/\r?\n/);let out="",code=false,buffer=[],list=false;
  const closeList=()=>{if(list){out+="</ul>";list=false;}};
  for(let i=0;i<lines.length;i++){
    const line=lines[i];
    if(line.startsWith("```")){closeList();if(code){out+=`<pre><code>${esc(buffer.join("\n"))}</code></pre>`;buffer=[];}code=!code;continue;}
    if(code){buffer.push(line);continue;}
    const heading=line.match(/^(#{1,3})\s+(.+)/);
    if(heading){closeList();const n=heading[1].length;out+=`<h${n}>${inline(heading[2])}</h${n}>`;continue;}
    if(/^\s*[-*]\s+/.test(line)){if(!list){out+="<ul>";list=true;}out+=`<li>${inline(line.replace(/^\s*[-*]\s+/,""))}</li>`;continue;}
    closeList();
    if(line.startsWith("|")&&i+1<lines.length&&/^\|[\s:|\-]+\|\s*$/.test(lines[i+1])){
      const cells=l=>l.trim().replace(/^\||\|$/g,"").split("|");
      out+="<table><thead><tr>"+cells(line).map(c=>`<th>${inline(c.trim())}</th>`).join("")+"</tr></thead><tbody>";i++;
      while(i+1<lines.length&&lines[i+1].startsWith("|")){i++;out+="<tr>"+cells(lines[i]).map(c=>`<td>${inline(c.trim())}</td>`).join("")+"</tr>";}out+="</tbody></table>";continue;
    }
    if(line.startsWith(">")){out+=`<blockquote>${inline(line.replace(/^>\s?/,""))}</blockquote>`;continue;}
    if(/^---+$/.test(line)){out+="<hr>";continue;}
    if(line.trim())out+=`<p>${inline(line)}</p>`;
  }
  closeList();if(buffer.length)out+=`<pre><code>${esc(buffer.join("\n"))}</code></pre>`;return out;
}
async function showArtifact(path) {
  const run=state.run, navigation=state.navigation, request=++state.artifactRequest;
  path=state.detail.writing.artifacts?.[path]||path;
  const result=await api(`/api/runs/${encodeURIComponent(run)}/artifact?path=${encodeURIComponent(path)}`);
  if(navigation!==state.navigation||request!==state.artifactRequest)return;
  state.selectedFile=path;const el=$("#artifact-dialog");el.className="artifact-dialog";
  el.innerHTML=`<div class="dialog-header"><div><h2 id="artifact-title">${esc(path.split("/").at(-1))}</h2><p>${esc(path)}</p></div><button class="btn ghost icon-only" data-close aria-label="Close document">${icon("close")}</button></div><article class="artifact-content">${path.endsWith(".md")?markdown(result.text):`<pre>${esc(result.text)}</pre>`}</article>`;
  if(!el.open)el.showModal();
}
async function updateLog() {
  if(!state.job)return;
  const selected=state.job, offset=state.offset, result=await api(`/api/jobs/${selected}/log?offset=${offset}`);
  if(state.job!==selected||state.offset!==offset)return;
  state.offset=result.offset;state.log+=result.text;
  if(state.log.length>300000)state.log="[Showing recent logs only; full logs are saved locally]\n"+state.log.slice(-250000);
  const el=$("#job-log");if(el){const bottom=el.scrollHeight-el.scrollTop-el.clientHeight<40;el.textContent=state.log||"Waiting for task output…";if(bottom)el.scrollTop=el.scrollHeight;}
}
async function poll() {
  if(state.polling||!state.run)return;
  state.polling=true;
  try{
    const id=state.run, navigation=state.navigation, next=await api(`/api/runs/${encodeURIComponent(id)}`);if(state.run!==id||navigation!==state.navigation)return;
    state.offline=false;
    const prev=state.detail, changed=JSON.stringify(next)!==JSON.stringify(prev);
    state.detail=next;
    if(!state.job&&next.jobs.length)state.job=next.jobs[0].id;
    if(changed&&!$("#modal").open&&!$("#artifact-dialog").open&&document.activeElement?.id!=="file-search")renderMain();
    await updateLog();
    const prior=prev?.jobs.find(j=>j.id===state.job), now=next.jobs.find(j=>j.id===state.job);
    if(prior?.status==="running"&&now&&now.status!=="running")toast(`${now.label}: ${statusLabels[now.status]}`,now.status==="failed"||now.status==="attention");
  }catch(error){if(!state.offline){toast("Connection to the local server was lost. Saved research files are preserved.",true);state.offline=true;}}finally{state.polling=false;}
}
document.addEventListener("click",async e=>{
  const b=e.target.closest("button, a");if(!b||b.disabled)return;
  try{
    if(b.hasAttribute("data-close")){b.closest("dialog").close();return;}
    if(b.dataset.file){await showArtifact(b.dataset.file);return;}
    if(b.dataset.action){showAction(b.dataset.action,JSON.parse(b.dataset.preset||"{}"));return;}
    if(b.dataset.delete){confirmDelete(b.dataset.delete);return;}
    if(b.dataset.run){await loadRun(b.dataset.run,b.dataset.targetTab||(!state.run?state.tab:"overview"));return;}
    if(b.dataset.tab){selectTab(b.dataset.tab);return;}
    if(b.dataset.job){state.job=b.dataset.job;state.offset=0;state.log="";renderMain();await updateLog();return;}
    if(b.dataset.cancel){
      const jobId=b.dataset.cancel;
      dialog("Stop the current task","Existing experiments, failures, and test-set access records are preserved.",[],async()=>{await api(`/api/jobs/${jobId}/cancel`,{});await poll();renderMain();},"Stop task",'<p class="form-note">Active model calls or training will be interrupted. Incurred costs are not refunded. You can resume from the saved stage; incomplete confirmatory experiments still need review.</p>');return;
    }
    switch(b.dataset.do){
      case "new":newProject();break;
      case "demo":newProject(true);break;
      case "settings":settings();break;
      case "connection":connection();break;
      case "import":importProject();break;
      case "sample":{const r=await api("/api/sample",{});state.boot=await api("/api/bootstrap");await loadRun(r.id);break;}
      case "home":goHome();break;
      case "reload":location.reload();break;
    }
  }catch(error){toast(error.message,true);}
});
document.addEventListener("input",e=>{
  if(e.target.id==="file-search"){
    state.filter=e.target.value;
    updateFileSearch();
  }
});
document.addEventListener("change",e=>{
  if(e.target.id==="field-mode"){
    const field=$("#field-profile");if(field)field.value=e.target.value==="demo"?"image-classification.yaml":"digits-mlp.yaml";
  }
});
$("#modal").addEventListener("close",()=>{if(state.detail)renderMain();});
async function start(){
  try{
    state.boot=await api("/api/bootstrap");
    const id=localStorage.getItem("scholarloop.run");
    if(id&&state.boot.runs.some(r=>r.id===id))await loadRun(id);else shell();
    setInterval(poll,2500);
  }catch(error){$("#app").innerHTML=`<div class="boot">Cannot open the workspace<span>${esc(error.message)}</span><button class="btn" data-do="reload">Reload page</button></div>`;}
}
start();
