import os, re, subprocess, tempfile
from typing import TypedDict, List, Optional
from flask import Flask, request, jsonify, render_template_string
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it in Render Environment Variables.")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=api_key, temperature=0)

def clean_response(response):
    content = response.content
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content).strip()
    return str(content).strip()

def extract_code(text):
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*(.*?)```", text, re.I | re.S)
    return m.group(1).strip() if m else text.strip()

class VerilogState(TypedDict):
    user_input: Optional[str]; verilog_code: Optional[str]; testbench_code: Optional[str]
    compile_output: Optional[str]; simulation_output: Optional[str]; verification: Optional[str]
    final_output: Optional[str]; next_step: Optional[str]; retry_count: int; messages: List[BaseMessage]

@tool
def generate_verilog_code(user_input: str) -> str:
    """Generate correct Verilog HDL code for the user's task."""
    prompt = """You are an expert Verilog HDL designer.
USER TASK:
{task}
Generate the complete synthesizable Verilog design. Use standard Verilog-2001/2005 only, not SystemVerilog. Do not add functionality. Do not generate a testbench. Return only the Verilog source inside one code block. Internally check syntax, ports, widths, operators, assignments, semicolons and begin/end.""".format(task=user_input)
    return extract_code(clean_response(llm.invoke(prompt)))

def verilog_node(s):
    try:
        code = generate_verilog_code.invoke({"user_input": s["user_input"]})
        if not code: raise ValueError("Empty Verilog code")
        return {"verilog_code": code, "next_step": "testbench"}
    except Exception as e:
        return {"simulation_output": "Verilog generation error: " + str(e), "next_step": "manager"}

@tool
def generate_testbench(user_input: str, verilog_code: str) -> str:
    """Generate a matching Verilog testbench."""
    prompt = """You are an expert Verilog verification engineer.
USER TASK:
{task}
VERILOG DESIGN:
{code}
Generate a complete matching Verilog-2005 testbench. Use the exact module name, ports and widths. Instantiate the design, drive all inputs, test meaningful cases, use suitable delays, $display for results and $finish. Do not modify the design and do not use SystemVerilog. Return only testbench source inside one code block.""".format(task=user_input, code=verilog_code)
    return extract_code(clean_response(llm.invoke(prompt)))

def tb_node(s):
    try:
        tb = generate_testbench.invoke({"user_input": s["user_input"], "verilog_code": s["verilog_code"]})
        if not tb: raise ValueError("Empty testbench")
        return {"testbench_code": tb, "next_step": "simulation"}
    except Exception as e:
        return {"simulation_output": "Testbench generation error: " + str(e), "next_step": "manager"}

@tool
def simulate_verilog(verilog_code: str, testbench_code: str) -> str:
    """Compile and simulate Verilog using Icarus Verilog."""
    d = tempfile.mkdtemp(); design=os.path.join(d,"design.v"); tb=os.path.join(d,"testbench.v"); out=os.path.join(d,"simulation.out")
    open(design,"w").write(verilog_code); open(tb,"w").write(testbench_code)
    c = subprocess.run(["iverilog","-g2005","-o",out,design,tb], capture_output=True, text=True)
    if c.returncode != 0:
        return "SIMULATION STATUS:\nCOMPILE ERROR\n\nCOMPILER ERROR:\n"+c.stderr+"\n\nCOMPILER OUTPUT:\n"+c.stdout
    r = subprocess.run(["vvp",out], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return "SIMULATION STATUS:\nSIMULATION ERROR\n\nSIMULATION ERROR:\n"+r.stderr+"\n\nSIMULATION OUTPUT:\n"+r.stdout
    return "SIMULATION STATUS:\nPASS\n\nSIMULATION OUTPUT:\n"+r.stdout+"\n\nSIMULATION ERRORS:\n"+r.stderr

def sim_node(s):
    try:
        result=simulate_verilog.invoke({"verilog_code":s["verilog_code"],"testbench_code":s["testbench_code"]})
        return {"simulation_output":result,"verification":"PASS" if "SIMULATION STATUS:\nPASS" in result else "FAIL","next_step":"verification" if "SIMULATION STATUS:\nPASS" in result else "manager"}
    except Exception as e:
        return {"simulation_output":"Simulation error: "+str(e),"verification":"FAIL","next_step":"manager"}

@tool
def verify_verilog(user_input: str, verilog_code: str, testbench_code: str, simulation_output: str) -> str:
    """Verify the Verilog and actual simulation result."""
    prompt="""You are a senior Verilog verification engineer.
USER TASK:
{task}
VERILOG:
{code}
TESTBENCH:
{tb}
ACTUAL SIMULATION:
{sim}
Check task correctness, module/ports, testbench, compilation and actual output. Return exactly:
VERIFICATION: PASS or FAIL
REASON: <short reason>
SIMULATION_RESULT: <short reason>
CORRECTION: NONE
Never invent simulation results.""".format(task=user_input,code=verilog_code,tb=testbench_code,sim=simulation_output)
    return clean_response(llm.invoke(prompt))

def verify_node(s):
    try:
        v=verify_verilog.invoke({"user_input":s["user_input"],"verilog_code":s["verilog_code"],"testbench_code":s["testbench_code"],"simulation_output":s["simulation_output"]})
        return {"verification":v,"next_step":"final_output"}
    except Exception as e: return {"verification":"FAIL\nReason: "+str(e),"next_step":"final_output"}

@tool
def fix_verilog(user_input: str, verilog_code: str, testbench_code: str, simulation_output: str) -> str:
    """Analyze an actual Icarus error and return corrected Verilog and testbench."""
    prompt="""You are an expert Verilog debugging engineer.
USER TASK:
{task}
CURRENT VERILOG:
{code}
CURRENT TESTBENCH:
{tb}
ACTUAL ICARUS OUTPUT:
{sim}
Fix the actual error. Preserve the task. Use Verilog-2005 only. Keep module and ports consistent. Return exactly these labels and complete source, with no Markdown fences:
VERILOG_CODE:
<corrected Verilog>
TESTBENCH_CODE:
<corrected testbench>""".format(task=user_input,code=verilog_code,tb=testbench_code,sim=simulation_output)
    return clean_response(llm.invoke(prompt))

def extract_fixed(text):
    a=re.search(r"VERILOG_CODE:\s*(.*?)(?=TESTBENCH_CODE:)",text,re.I|re.S); b=re.search(r"TESTBENCH_CODE:\s*(.*)",text,re.I|re.S)
    if not a or not b: raise ValueError("Corrected code was not found")
    return extract_code(a.group(1)), extract_code(b.group(1))

def repair_node(s):
    try:
        text=fix_verilog.invoke({"user_input":s["user_input"],"verilog_code":s["verilog_code"],"testbench_code":s["testbench_code"],"simulation_output":s["simulation_output"]})
        code,tb=extract_fixed(text); n=s.get("retry_count",0)+1
        return {"verilog_code":code,"testbench_code":tb,"retry_count":n,"next_step":"simulation"}
    except Exception as e:
        return {"retry_count":s.get("retry_count",0)+1,"simulation_output":s.get("simulation_output","")+"\nRepair error: "+str(e),"next_step":"final_output"}

def final_node(s):
    result=("============================================================\n"
            "                 VERILOG SOLUTION\n"
            "============================================================\n\n"
            "VERILOG CODE:\n------------------------------------------------------------\n"+s.get("verilog_code","")+"\n\n"
            "TESTBENCH CODE:\n------------------------------------------------------------\n"+s.get("testbench_code","")+"\n\n"
            "SIMULATION RESULT:\n------------------------------------------------------------\n"+s.get("simulation_output","")+"\n\n"
            "VERIFICATION:\n------------------------------------------------------------\n"+s.get("verification","")+"\n\n"
            "============================================================")
    return {"final_output":result,"next_step":"end"}

def task_node(s):
    if not s.get("user_input"): raise ValueError("Verilog task is missing")
    return {"next_step":"verilog_generator"}

def manager_node(s):
    if s.get("next_step")=="manager": return {"next_step":"repair" if s.get("retry_count",0)<3 else "final_output"}
    return {"next_step":s.get("next_step","final_output")}

workflow=StateGraph(VerilogState)
workflow.add_node("task_input",task_node); workflow.add_node("verilog_generator",verilog_node); workflow.add_node("testbench",tb_node); workflow.add_node("simulation",sim_node); workflow.add_node("verification",verify_node); workflow.add_node("repair",repair_node); workflow.add_node("manager",manager_node); workflow.add_node("final_output",final_node)
workflow.add_edge(START,"task_input")
workflow.add_conditional_edges("task_input",lambda s:s["next_step"],{"verilog_generator":"verilog_generator"})
workflow.add_conditional_edges("verilog_generator",lambda s:s["next_step"],{"testbench":"testbench","manager":"manager"})
workflow.add_conditional_edges("testbench",lambda s:s["next_step"],{"simulation":"simulation","manager":"manager"})
workflow.add_conditional_edges("simulation",lambda s:s["next_step"],{"verification":"verification","manager":"manager"})
workflow.add_conditional_edges("verification",lambda s:s["next_step"],{"final_output":"final_output"})
workflow.add_conditional_edges("manager",lambda s:s["next_step"],{"repair":"repair","final_output":"final_output"})
workflow.add_conditional_edges("repair",lambda s:s["next_step"],{"simulation":"simulation","final_output":"final_output"})
workflow.add_conditional_edges("final_output",lambda s:s["next_step"],{"end":END})
verilog_app=workflow.compile()

HTML='''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Verilog Agent</title><style>body{font-family:Arial;background:#f4f6f8;margin:0;padding:30px}.box{max-width:1000px;margin:auto;background:white;padding:30px;border-radius:12px}textarea{width:100%;min-height:160px;box-sizing:border-box;padding:12px;font-size:16px}button{width:100%;padding:14px;margin-top:12px;background:#222;color:white;border:0;border-radius:8px;font-size:16px}#loading,#result{margin-top:20px}#loading{display:none}#result{display:none;background:#f7f7f7;padding:20px;white-space:pre-wrap;overflow:auto;font-family:Consolas,monospace}</style></head><body><div class="box"><h1>Verilog Code Generation Agent</h1><p>Generate Verilog + matching testbench + real Icarus simulation + verification.</p><textarea id="task" placeholder="Example: Design a 2-to-1 multiplexer using Verilog."></textarea><button onclick="runAgent()">Generate & Simulate</button><div id="loading">Generating and simulating... Please wait.</div><div id="result"></div></div><script>async function runAgent(){let t=document.getElementById('task').value,r=document.getElementById('result'),l=document.getElementById('loading');if(!t.trim()){alert('Please enter a Verilog design task.');return}r.style.display='none';l.style.display='block';try{let x=await fetch('/agent/playground/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:t})});let d=await x.json();r.textContent=d.success?d.result:'ERROR:\n'+d.error}catch(e){r.textContent='ERROR:\n'+e.message}l.style.display='none';r.style.display='block'}}</script></body></html>'''

@app.route("/")
def home(): return '<h2>Verilog Code Generation Agent</h2><p>Open <a href="/agent/playground/">/agent/playground/</a></p>'
@app.route("/agent/playground/")
def playground(): return render_template_string(HTML)
@app.route("/agent/playground/analyze",methods=["POST"])
def analyze():
    try:
        data=request.get_json(silent=True) or {}; task=data.get("task","").strip()
        if not task: return jsonify(success=False,error="Verilog design task is required."),400
        state={"messages":[HumanMessage(content=task)],"user_input":task,"verilog_code":"","testbench_code":"","compile_output":"","simulation_output":"","verification":"","final_output":"","next_step":"verilog_generator","retry_count":0}
        result=verilog_app.invoke(state)
        return jsonify(success=True,result=result.get("final_output","No final output generated."))
    except Exception as e:
        traceback.print_exc(); return jsonify(success=False,error=str(e)),500

if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.getenv("PORT","10000")))
