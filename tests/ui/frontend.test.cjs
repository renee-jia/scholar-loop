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
