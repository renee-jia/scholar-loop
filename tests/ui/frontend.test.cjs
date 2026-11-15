// Run with: node --test tests/ui/frontend.test.cjs
// Exercise UI state transitions without a browser, network, or model requests.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../../scholarloop/ui/static/app.js"), "utf8")
  .replace(/\nstart\(\);\s*$/, "\n");

function element() {
  const classes = new Set();
  return {innerHTML:"", textContent:"", hidden:true, open:false, attributes:{},
    classList:{toggle(name,on){if(on)classes.add(name);else classes.delete(name);}, contains:name=>classes.has(name)},
    setAttribute(name,value){this.attributes[name]=value;}, focus(){this.focused=true;},
    addEventListener(){}, showModal(){this.open=true;}};
}

function detail(id="a") {
  return {config:{id,title:id,mode:"live",readonly:false,profile:"digits-mlp.yaml",spend_cap:10,max_calls:100},
    session:{stage:"new",ideas:[]}, review:{status:null,issues:[],reports:[],release_error:""},
    writing:{stage:"new",artifacts:{}}, handoff:{}, jobs:[], entries:[], protocols:[], artifacts:[], warnings:[], spend:{}};
}

function harness() {
  const nodes = new Map(), storage = new Map(), handlers = {};
  const node = selector => {
    if(!nodes.has(selector))nodes.set(selector,element());
    return nodes.get(selector);
  };
  node('meta[name="scholarloop-token"]').content="test-token";
  const context = vm.createContext({
    document:{querySelector:node,addEventListener:(event,fn)=>handlers[event]=fn,activeElement:null},
    localStorage:{setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k),getItem:k=>storage.get(k)},
    window:{scrollTo(){}},setTimeout:()=>1,clearTimeout(){},setInterval(){},
    fetch:async()=>{throw new Error("Unexpected network request");},
  });
  const run = script => vm.runInContext(script,context);
  run(source);
  context.fixture=detail();
  run('state.detail=fixture;state.run="a";state.boot={runs:[],profiles:[],actions:[],environment:{default_model:"test"}};');
  return {run,node,storage,handlers,context,state:run("state")};
}

function pendingRequests(h) {
  const requests=[];
  h.context.fetch=(url)=>new Promise(resolve=>requests.push({url,finish:data=>resolve({ok:true,json:async()=>data})}));
  return requests;
}

test("next-step actions open forms even when an action shares a tab name",()=>{
  const h=harness();
  h.state.detail.session={stage:"reviewed",direction_map:[{id:"d1"}],ideas:[]};
  h.state.detail.artifacts=[{path:"lit_review.md"}];
  assert.match(h.run("callout()"),/data-action="ideas"/);
  assert.doesNotMatch(h.run("callout()"),/data-tab="ideas"/);
  h.state.detail.session.stage="selected";
  h.state.detail.entries=[{score:1}];
  assert.match(h.run("callout()"),/data-action="review"/);
});

test("busy demo and archived research still offer working navigation",()=>{
  const h=harness();
  h.state.detail.config.mode="demo";
  h.state.detail.jobs=[{status:"running"}];
  assert.match(h.run("callout()"),/data-tab="experiments"/);
  assert.doesNotMatch(h.run("callout()"),/disabled/);
  h.state.detail.config.readonly=true;
  assert.match(h.run("callout()"),/data-tab="files"/);
});

test("CLI research without an ideation session resumes from its existing evidence",()=>{
  const h=harness();
  h.state.detail.session={};
  h.state.detail.entries=[{score:1}];
  assert.match(h.run("callout()"),/data-action="review"/);
  h.state.detail.review.status="READY_FOR_AUTHOR";
  assert.match(h.run("callout()"),/data-action="approve"/);
  h.state.detail.review.status="READY_TO_WRITE";
  assert.match(h.run("callout()"),/data-action="write"/);
});

test("reading the approval report cannot submit its surrounding form",()=>{
  const h=harness();
  const html=h.run('artifactButton("scientific_review.md","Read report")');
  assert.match(html,/<button type="button"/);
  assert.match(html,/data-file="scientific_review.md"/);
  assert.doesNotMatch(html,/type="submit"|data-action=/);
});

test("unavailable actions explain prerequisites and stale approvals block writing",()=>{
  const h=harness();
  assert.match(h.run('button("campaign","Start")'),/disabled title="Select a research idea first/);
  h.state.detail.handoff={idea_id:"i1"};
  assert.doesNotMatch(h.run('button("campaign","Start")'),/disabled/);
  assert.match(h.run('button("write","Write")'),/disabled title="Complete scientific review/);
  h.state.detail.review.status="READY_TO_WRITE";
  assert.doesNotMatch(h.run('button("write","Write")'),/disabled/);
  h.state.detail.review.release_error="Evidence changed";
  assert.match(h.run('button("write","Write")'),/disabled/);
  h.state.detail.review.release_error="";
  h.state.detail.writing.stage="reviewed";
  assert.doesNotMatch(h.run('button("write_feedback","Revise")'),/disabled/);
});

test("archive and running-job settings cannot open an editable form",()=>{
  const h=harness();
  h.state.detail.config.readonly=true;
  assert.match(h.run("overview()"),/data-do="settings" disabled title="This archive is read-only/);
  h.run("settings()");
  assert.equal(h.node("#modal").open,false);
  h.state.detail.config.readonly=false;
  h.state.detail.jobs=[{id:"j",status:"running",label:"Experiment",started:1}];
  assert.match(h.run("overview()"),/data-do="settings" disabled title="Wait/);
});

test("filtering artifacts updates the count and displayed rows together",()=>{
  const h=harness();
  h.state.detail.artifacts=[{path:"paper.md",size:40},{path:"review.md",size:20}];
  h.state.filter="paper";
  h.run("updateFileSearch()");
  assert.equal(h.node("#file-count").textContent,"1 of 2 research files");
  assert.match(h.node("#file-list").innerHTML,/paper\.md/);
  assert.doesNotMatch(h.node("#file-list").innerHTML,/review\.md/);
  h.state.filter="missing";
  h.run("updateFileSearch()");
  assert.equal(h.node("#file-count").textContent,"0 of 2 research files");
});

test("switching projects clears the previous artifact filter",async()=>{
  const h=harness(), requests=pendingRequests(h);
  h.state.filter="paper";
  const navigation=h.run('loadRun("b")');
  requests[0].finish(detail("b"));
  await navigation;
  assert.equal(h.state.run,"b");
  assert.equal(h.state.filter,"");
});

test("the last selected project wins when responses arrive out of order",async()=>{
  const h=harness(), requests=pendingRequests(h);
  const earlier=h.run('loadRun("b")'), later=h.run('loadRun("c")');
  requests[1].finish(detail("c"));await later;
  requests[0].finish(detail("b"));await earlier;
  assert.equal(h.state.run,"c");
  assert.equal(h.state.detail.config.id,"c");
  assert.equal(h.storage.get("scholarloop.run"),"c");
});

test("returning home cancels an in-flight project navigation",async()=>{
  const h=harness(), requests=pendingRequests(h);
  const loading=h.run('loadRun("b")');
  h.run("goHome()");
  requests[0].finish(detail("b"));await loading;
  assert.equal(h.state.run,null);
  assert.equal(h.state.detail,null);
  assert.equal(h.storage.has("scholarloop.run"),false);
});

test("a delayed document response does not open over another page",async()=>{
  const h=harness(), requests=pendingRequests(h);
  const reading=h.run('showArtifact("paper.md")');
  h.run('selectTab("experiments")');
  requests[0].finish({text:"Old document"});await reading;
  assert.equal(h.node("#artifact-dialog").open,false);
});

test("overlapping log reads cannot append the same output twice",async()=>{
  const h=harness(), requests=pendingRequests(h);
  h.state.job="job";
  const first=h.run("updateLog()"),second=h.run("updateLog()");
  requests[1].finish({offset:5,text:"hello"});await second;
  requests[0].finish({offset:5,text:"hello"});await first;
  assert.equal(h.state.log,"hello");
  assert.equal(h.state.offset,5);
});

test("All research keeps workflow tabs enabled and asks which project to open",()=>{
  const h=harness();
  h.state.boot.runs=[{...detail("b").config,created:1}];
  h.run("goHome()");
  assert.doesNotMatch(h.node("#app").innerHTML,/data-tab="(?:experiments|writing)" disabled/);
  h.run('selectTab("experiments")');
  assert.equal(h.state.run,null);
  assert.equal(h.state.tab,"experiments");
  assert.match(h.node("#main").innerHTML,/Choose a research project/);
  assert.match(h.node("#main").innerHTML,/data-target-tab="experiments"/);
  h.run('selectTab("writing")');
  assert.match(h.node("#main").innerHTML,/data-target-tab="writing"/);
});

test("choosing a project opens the workflow page the user requested",async()=>{
  const h=harness(), requests=pendingRequests(h);
  h.run("goHome()");
  h.run('selectTab("writing")');
  const b={dataset:{run:"b",targetTab:"writing"},disabled:false,hasAttribute:()=>false};
  const click=h.handlers.click({target:{closest:()=>b}});
  requests[0].finish(detail("b"));await click;
  assert.equal(h.state.run,"b");
  assert.equal(h.state.tab,"writing");
});

test("deletion makes no request until the confirmation callback runs",async()=>{
  const h=harness(), calls=[];
  h.state.boot.runs=[{...h.state.detail.config,created:1}];
  let confirmation;
  h.context.captureDialog=(...args)=>{confirmation=args;};
  h.run("dialog=captureDialog");
  h.context.fetch=async(url,options)=>{
    calls.push({url,body:options?.body});
    return {ok:true,json:async()=>url==="/api/bootstrap"?{...h.state.boot,runs:[]}:{deleted:true}};
  };
  h.run('confirmDelete("a")');
  assert.equal(calls.length,0);
  assert.equal(confirmation[0],"Delete research?");
  assert.equal(confirmation[4],"Confirm deletion");
  await confirmation[3]();
  assert.equal(calls[0].url,"/api/runs/a/delete");
  assert.deepEqual(JSON.parse(calls[0].body),{confirmed:true});
  assert.equal(h.state.run,null);
  assert.equal(h.state.boot.runs.length,0);
});

test("an archive shows what it recorded, never an invitation to act",()=>{
  const h=harness();
  h.state.detail.config.readonly=true;
  h.state.detail.entries=[{id:"e",score:1,fidelity:["full"]}];
  h.state.detail.artifacts=[{path:"ideas.md",size:1}];
  h.state.detail.writing.artifacts={"review.md":"review.md"};
  assert.match(h.run("ideas()"),/data-file="ideas.md"/);
  assert.doesNotMatch(h.run("ideas()"),/Develop a testable hypothesis/);
  assert.match(h.run("review()"),/predates the scientific review/);
  assert.match(h.run("research()"),/Nothing was recorded for this stage/);
  assert.match(h.run("stats()"),/Recorded in this run&#39;s README|Recorded in this run's README/);
  assert.doesNotMatch(h.run("stats()"),/\$0\.00/);
  h.state.detail.review.status="READY_TO_WRITE";
  h.state.detail.writing.stage="reviewed";
  assert.equal((h.run("progress()").match(/workflow-step done/g)||[]).length,5);
});

test("a finished study is marked ready, and the recorded runs are one click away",()=>{
  const h=harness();
  assert.doesNotMatch(h.run("callout()"),/callout ready/);
  h.state.detail.review.status="READY_TO_WRITE";
  h.state.detail.writing.stage="reviewed";
  assert.match(h.run("callout()"),/class="callout ready"/);
  h.state.detail.review.status=null;
  h.state.detail.config.readonly=true;
  assert.match(h.run("callout()"),/class="callout ready"/);
  assert.match(h.run("home()"),/class="btn" data-do="sample"/);
});

test("an idea leads with its hypothesis; the chosen one is marked and unfolded",()=>{
  const h=harness();
  const idea=n=>({idea_id:"i"+n,title:"T"+n,description:"First step. Second step.",hypothesis:{claim:"Claim "+n,predicted_effect:"One point lower."},
    experiment_plan:"Train five seeds. Compare both arms.",sources:["arXiv:1"],feasibility:4,impact:3,tournament:{elo:1214.5,rank:n,wins:2,losses:1,draws:1},novelty_review:{status:"provisional_difference"}});
  h.state.detail.session.ideas=[idea(1),idea(2)];
  h.state.detail.session.selected=2;
  const html=h.run("ideas()");
  assert.match(html,/class="cards idea-list"/);
  const cards=html.split("<article").slice(1);
  assert.equal(cards.length,2);
  assert.doesNotMatch(cards[0],/idea-card chosen/);
  assert.match(cards[1],/idea-card chosen/);
  assert.match(cards[0],/idea-hypothesis"><div class="idea-label">Hypothesis<\/div><p>Claim 1<\/p>/);
  assert.match(cards[0],/Elo 1215/);
  assert.match(cards[0],/2W · 1L · 1D/);
  assert.match(cards[0],/Feasibility .*4\/5/);
  assert.match(cards[0],/provisional difference/);
  assert.match(cards[0],/<details class="idea-fold"><summary>Experiment plan/);
  assert.match(cards[1],/<details class="idea-fold" open><summary>Experiment plan<\/summary><ul class="points"><li>Train five seeds\.<\/li><li>Compare both arms\.<\/li>/);
});

test("a long recorded paragraph is shown one sentence per point, words unchanged",()=>{
  const h=harness();
  const text="All setup statements match the code, e.g. the lr of 0.003. Probe trajectories are identical. No issue remains.";
  const html=h.run(`points(${JSON.stringify(text)})`);
  assert.equal((html.match(/<li>/g)||[]).length,3);
  assert.match(html,/e\.g\. the lr of 0\.003\.<\/li>/);
  assert.equal(html.replace(/<[^>]+>/g," ").replace(/\s+/g," ").trim(),text);
  assert.match(h.run(`points("One sentence only.")`),/^<p>/);
});

test("artifacts are grouped by stage, each file once, only the first group unfolded until searched",()=>{
  const h=harness();
  const registry=Array.from({length:20},(_,i)=>({path:`registry/exp_${i}.json`,size:10}));
  const files=[{path:"paper.md",size:1},{path:"scientific_review.md",size:1},{path:"review.md",size:1},{path:"lit_review.md",size:1},
    {path:"reliability/claims.md",size:1},{path:"README.md",size:1},{path:"experiments.jsonl",size:1},...registry];
  const html=h.run(`fileRows(${JSON.stringify(files)})`);
  const titles=[...html.matchAll(/file-group-title">([^<]+)</g)].map(m=>m[1]);
  assert.deepEqual(titles,["Paper and writing","Scientific review","Experiments","Literature and ideas","Reliability checks","Other"]);
  assert.equal((html.match(/class="list-row"/g)||[]).length,files.length);
  const groups=html.split("<details").slice(1);
  assert.match(groups[0],/data-file="paper.md"/);
  assert.match(groups[0],/data-file="review.md"/);
  assert.match(groups[1],/data-file="scientific_review.md"/);
  assert.match(groups[0].split("<summary>")[0],/open/);
  assert.doesNotMatch(groups[1].split("<summary>")[0],/open/);
  assert.doesNotMatch(groups[2].split("<summary>")[0],/open/);
  h.state.filter="exp_1";
  assert.match(h.run(`fileRows(${JSON.stringify(registry)})`).split("<summary>")[0],/open/);
});

test("the review page leads with the verdict and its counts; open issues come first, unfolded",()=>{
  const h=harness();
  const report=(role,verdict)=>({role,verdict,summary:"One. Two. Three. Four. Five.",claim_checks:[{status:"supported_in_scope"},{status:"supported_in_scope"}]});
  h.state.detail.review={status:"READY_TO_WRITE",hash:"abc",release_error:"",reports:[report("implementation","pass"),report("argument","pass")],
    issues:[{id:"a",target:"claim_a",severity:"minor",status:"noted",role:"argument",action:"reframe",observation:"Seen."},
            {id:"b",target:"claim_b",severity:"major",status:"verified",role:"implementation",action:"inspect",observation:"Seen.",resolution:{reason:"Checked the logs."}}]};
  let html=h.run("review()");
  assert.match(html,/review-verdict approved/);
  assert.match(html,/<b>2<small>\/2<\/small><\/b><span>Reviewers passed/);
  assert.match(html,/<b>4<small>\/4<\/small><\/b><span>Claim checks supported/);
  assert.match(html,/<b>0<\/b><span>Issues still open/);
  assert.match(html,/<b>2\/2<\/b> claims supported<\/span><span><b>1<\/b> issues raised/);
  assert.match(html,/2 more points/);
  assert.doesNotMatch(html,/file-group-title">Open</);
  assert.ok(html.indexOf("Major, closed")<html.indexOf("Minor, closed"));
  assert.match(html,/How it was closed<\/div><p>Checked the logs\./);
  h.state.detail.review.status="NEEDS_REVISION";
  h.state.detail.review.issues[0].status="open";
  html=h.run("review()");
  assert.match(html,/review-verdict waiting/);
  assert.match(html,/issue-group" open><summary><span class="file-group-title">Open</);
  assert.match(html,/review-stat alert"><b>1<\/b><span>Issues still open/);
});

test("recognised sources link to their public pages; anything else stays plain text",()=>{
  const h=harness();
  assert.equal(h.run('sourceLink("arXiv:2306.13649")'),"https://arxiv.org/abs/2306.13649");
  assert.equal(h.run('sourceLink("arXiv:2106.10885v1")'),"https://arxiv.org/abs/2106.10885v1");
  assert.equal(h.run('sourceLink("doi:10.1007/s00521-025-11162-0")'),"https://doi.org/10.1007/s00521-025-11162-0");
  assert.equal(h.run('sourceLink("openalex:W3007689282")'),"https://openalex.org/W3007689282");
  assert.equal(h.run('sourceLink("arXiv:x\\" onclick=1")'),"");
  assert.equal(h.run('sourceLink("javascript:alert(1)")'),"");
  assert.match(h.run('sourceChip("arXiv:2306.13649")'),/^<a class="source-chip" href="https:\/\/arxiv\.org\/abs\/2306\.13649" target="_blank" rel="noopener noreferrer"/);
  assert.match(h.run('sourceChip("my own notes")'),/^<span>my own notes<\/span>$/);
});

test("the writing page leads with the manuscript and marks each stage done, current or still to come",()=>{
  const h=harness();
  h.state.detail.review.status="READY_TO_WRITE";
  h.state.detail.writing={stage:"keypointed",artifacts:{"outline.md":"outline.md","key_points.md":"key_points.md"},summary:{title:"A <b>title</b>",sections:0,words:0,reviewers:0,mean_score:null}};
  let html=h.run("writing()");
  assert.match(html,/Manuscript · stage 3 of 4/);
  assert.match(html,/A &lt;b&gt;title&lt;\/b&gt;/);
  assert.deepEqual([...html.matchAll(/write-step (done|current|todo)/g)].map(m=>m[1]),["done","done","current","todo"]);
  assert.match(html,/Up next/);
  h.state.detail.writing={stage:"reviewed",artifacts:{"paper.md":"paper.md","paper_traced.md":"paper_traced.md","evidence_map.md":"evidence_map.md"},summary:{title:"T",sections:8,words:3376,reviewers:3,mean_score:2.2,recommendation:"reject"}};
  html=h.run("writing()");
  assert.match(html,/manuscript-hero done/);
  assert.match(html,/<b>3,376<\/b><span>Words/);
  assert.match(html,/<b>2\.2<small>\/10<\/small><\/b><span>Panel score · reject/);
  assert.match(html,/data-file="paper_traced.md"/);
  assert.doesNotMatch(html,/data-action="write"/);
});

test("tools are shelved by whether they call the model",()=>{
  const h=harness();
  h.state.boot.actions=["evaluate","baseline","compare","traces","new_tool"].map(id=>({id,label:id,group:"tools"}));
  const html=h.run("toolsPage()").split('<div class="spacer">')[0], [free,live]=html.split('<section class="tool-shelf').slice(1);
  assert.deepEqual([...free.matchAll(/<h3>([^<]+)</g)].map(m=>m[1]),["evaluate","traces"]);
  assert.deepEqual([...live.matchAll(/<h3>([^<]+)</g)].map(m=>m[1]),["baseline","compare","new_tool"]);
  assert.match(live,/Random search alone is free/);
});

test("full results are bars from zero, one per run, named without the shared protocol prefix",()=>{
  const h=harness();
  h.state.detail.entries=[{id:"protocol_8198ab82d8c25b04_off_policy_ref",score:14,fidelity:["full"]},{id:"protocol_8198ab82d8c25b04_pure_on_policy",score:7,fidelity:["full"]},{id:"x",score:1,fidelity:["probe"]}];
  const html=h.run("chart()");
  assert.equal((html.match(/class="chart-bar"/g)||[]).length,2);
  assert.match(html,/>off_policy_ref<\/text>/);
  assert.match(html,/class="chart-bar" x="190" y="6" width="320\.0"/);
  assert.match(html,/class="chart-bar" x="190" y="36" width="160\.0"/);
  assert.equal(h.run('shortId("exp_0007")'),"exp_0007");
});

test("a direction leads with its open question; an archive with no tasks shows no task history",()=>{
  const h=harness();
  h.state.detail.session={stage:"mapped",chosen_directions:["d2"],ideas:[]};
  const cards=[1,2].map(n=>h.run(`directionCard({id:"d${n}",name:"N${n}",stance:"challenges",thesis:"One. Two.",open_question:"Why?",first_experiment:"Run it."},${n-1})`));
  assert.match(cards[0],/Direction 01 · Challenges your hypothesis/);
  assert.match(cards[0],/idea-label">Open question<\/div><p>Why\?<\/p>/);
  assert.doesNotMatch(cards[0],/idea-card chosen/);
  assert.match(cards[1],/idea-card chosen/);
  assert.match(h.run("jobPanel()"),/Task history/);
  h.state.detail.config.readonly=true;
  assert.equal(h.run("jobPanel()"),"");
});
