import json
import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END

# LLM uses delimiters in the tags to separate ideas.
# . any single character; *? zero or more characters; note: if there is no full stop, line breaks are also included
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

ALLOWED_TOOLS = ("search_evidence", "analyze_sentiment")
MAX_STEPS = 4                                           # 4 Planning steps
EVIDENCE_CHARS = 300
MAX_REVISIONS = 1                                       # 1 rectification

def clean_think(text):
    return THINK_RE.sub("", text or "").strip()         # removes <think> and </think> tags ... concerns qwen/reasoning

def extract_json(text):
    text = clean_think(text)                                                   
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()     #  Removes Markdown code delimiters ''' (and JSON if present)
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):                         #  LLM response from object {"verdict": "APPROVED"} or list [{"tool": ...}, {"tool": ...}]
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append((start, text[start:end + 1]))                 # Opening bracket, closing bracket, and opening bracket follows > ...relevant text
    candidates.sort(key=lambda c: c[0])                                     # Sort candidates by calculated value
    for position, fragment in candidates:                                   # _ Start position (not required) ...just a JSON snippet – fragment = JSON text
        try:
            return json.loads(fragment)
        except Exception:
            continue
    return None

# GRAPH STATE, AgentState / Agent status – the agent’s memory 
# ...a form that goes through all the stages or steps
class AgentState(TypedDict, total=False):   # total false ...Not every field needs to be filled in.
    goal: str
    plan: List[Dict[str, str]]
    trace: List[str]
    evidence: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    recommendation: str
    validation: Dict[str, Any]
    revisions: int                      


# Newly found documents added to the pool (which assigns unique identifiers (ID) and prevents duplicates)
def add_evidence(evidence, hits):
    #Append hits to the global evidence pool (dedup by URL/title); return their ids.
    ids = []
    for title, text, source, url, score in hits:
        key = url or title
        existing = None
        for e in evidence:
            if e["url"]:
                identifier = e["url"]
            else:
                identifier = e["title"]
            # Compare with the key you are looking for
            if identifier == key:
                existing = e
                break 
        # If nothing is found, `existing` remains `None`
        if existing:
            ids.append(existing["id"])
            continue
        ev = {"id": len(evidence) + 1, "title": title, "text": text,
              "source": source, "url": url, "score": round(float(score), 3)}
        evidence.append(ev)
        ids.append(ev["id"])
    return ids

# Converts the document pool (List Dictionaries) into a numeric string (for an LLM prompt)
def evidence_block(evidence):
    lines = []
    for e in evidence:
        truncated_text = e["text"][:EVIDENCE_CHARS]
        line = "Evidence {0}: {1} (Source: {2} - {3})".format(
            e["id"],            # -> {0}
            truncated_text,     # -> {1}
            e["source"],        # -> {2}
            e["url"])           # -> {3}
        lines.append(line)
    return "\n".join(lines)

def cited_ids(recommendation):
    # Search for the text following the section "SUPPORTING DOCUMENTS:"
    # Find the exact text "SUPPORTING DOCUMENTS:"
    # (.*) captures everything that follows on the same line and stores it as "Group 1"
    # re.I (= re.IGNORECASE) means: case-insensitive
    m = re.search(r"SUPPORTING DOCUMENTS:(.*)", recommendation, re.I)

    # Step 2: Check whether anything was found at all.
    # re.search returns a match object if it finds something,
    # otherwise None.
    if m:
        # m.group(1) is the captured group from (.*), i.e. everything after "SUPPORTING DOCUMENTS:"
        found_text = m.group(1)

        number_strings = re.findall(r"\d+", found_text)

        result = []
        for n in number_strings:
            result.append(int(n))
        return result
    else:
        return []
    
#---->----> Build and compile the LangGraph agent <---<---
# LLM, search and sentiment are fed in from outside. This means that the CLI and the dashboard share the same agent, 
# without having to load the models twice.
def build_agent(llm, search, sentiment_of_texts, analyze=None):

    # Helper: sends a text to the LLM (llm.invoke) and returns the cleaned-up response.
    def ask(prompt):
        # Send the question (prompt) to the language model
        # llm.invoke(prompt) means:
        # llm is the ChatOllama model via LangChain
        # .invoke(...) starts ONE run: in goes the question, out comes the answer.
        response = llm.invoke(prompt)
        # Extract plain text from a reply
        # The model does not return a string directly, but rather a message object (e.g. AIMessage). The actual text is contained in the .content attribute
        raw_text = response.content
        clean_text = clean_think(raw_text)
        return clean_text
    
    # Plan – The agent asks the LLM for a plan in the form of a JSON list containing 2–4 steps
    # Each step includes a tool (search_evidence or analyse_sentiment), a query (search query) and a reason (justification)
    def plan_node(state: AgentState):
        goal = state["goal"]
        prompt = (
            "You are the PLANNING module of an autonomous strategy agent for Lufthansa.\n"
            "Goal: " + goal + "\n\n"
            "Available tools:\n"
            "- search_evidence(query): semantic search over a news/RAG knowledge base\n"
            "- analyze_sentiment(query): retrieves docs for a topic and measures public sentiment\n\n"
            "Decide which tools to use. Produce a focused PLAN of 2 to 4 steps.\n"
            'Return ONLY a JSON array. Each element: '
            '{"tool": "search_evidence" or "analyze_sentiment", '
            '"query": "<focused English search query>", "reason": "<why this step>"}\n'
            "No prose, no markdown, JSON only."
        )
        steps = extract_json(ask(prompt))
        plan = []
        if isinstance(steps, list):
            for s in steps:
                # Condition 1: Is s even a dictionary?
                is_dictionary = isinstance(s, dict)
                # Condition 2: Is the specified tool permitted?
                if is_dictionary:
                    tool_allowed = s.get("tool") in ALLOWED_TOOLS
                else:
                    tool_allowed = False
                # Condition 3: Is there even a query (search request)?
                if is_dictionary:
                    has_query = bool(s.get("query"))
                else:
                    has_query = False
                if is_dictionary and tool_allowed and has_query:
                    # A new, clean dictionary
                    new_step = {
                        "tool": s["tool"],            # das Tool (jetzt sicher vorhanden)
                        "query": str(s["query"]),     # query, sicherheitshalber in Text umgewandelt
                        "reason": str(s.get("reason", ""))  # Begründung, falls vorhanden
                    }
                    plan.append(new_step)
        #If the LLM does not provide a viable plan, a fixed fallback plan is triggered – the demo will always remain operational
        if not plan: # A deterministic contingency plan to ensure the demo never crashes
            """
            The plan comes from the LLM – and LLMs are unreliable. The following may occur:
            - the model does not return valid JSON (→ _extract_json returns None),
            - it returns nonsense (an unauthorised tool, an empty query),
            - the response is completely empty.

            In all these cases, the plan remains empty ([]).   
            """
            plan = [
                {"tool": "search_evidence", "query": goal,
                 "reason": "direct retrieval for the goal"},
                {"tool": "search_evidence", "query": "biggest risks and threats for Lufthansa",
                 "reason": "risk scan"},
                {"tool": "analyze_sentiment", "query": "Lufthansa public perception",
                 "reason": "sentiment context"},
            ]
        return {"plan": plan[:MAX_STEPS], "evidence": [], "trace": [],
                "sentiment": {}, "revisions": 0}

    # Go through the plan step by step and open the tools
    def execute_node(state: AgentState):
        evidence, trace, sentiment = [], [], {}
        # Every result is added to the evidence pool via _add_evidence; every action is logged in the trace
        for i, step in enumerate(state["plan"], 1):
            tool, query = step["tool"], step["query"]
            if tool == "analyze_sentiment":
                # For `analyze_sentiment`: Retrieve 8 documents and measure the sentiment
                if analyze is not None:
                    # use the injected analyze_sentiment @tool (CLI)
                    dist, hits = analyze(query, 8)
                else:
                    # fallback if no analyze tool was provided (dashboard)
                    hits = search(query, 8)
                    dist = sentiment_of_texts([h[1] for h in hits])
                ids = add_evidence(evidence, hits)
                sentiment[query] = dist
                trace.append(f"Step {i}: analyze_sentiment(query='{query}') "
                             f"-> {dist}; evidence {ids}")
            else:
                # Other (search_evidence): Retrieve 5 documents
                hits = search(query, 5)
                ids = add_evidence(evidence, hits)
                trace.append(f"Step {i}: search_evidence(query='{query}') "
                             f"-> {len(hits)} docs; evidence {ids}")
        return {"evidence": evidence, "trace": trace, "sentiment": sentiment}
    
    # Creates a prompt based on the numbered documents and the measured sentiment, and has the LLM write a structured recommendation
    def decide_node(state: AgentState):
        goal = state["goal"]
        evidence = state["evidence"]
        sentiment_note = ""
        if state.get("sentiment"):
            sentiment_note = "Measured public sentiment (by topic): " + str(state["sentiment"]) + "\n"
 

        feedback = ""
        prev = state.get("validation")
        if prev and prev.get("verdict") == "NEEDS_REVISION":
            feedback = ("A reviewer flagged these issues with your previous draft; fix them and "
                        "cite only existing evidence numbers:\n- "
                        + "\n- ".join(prev.get("issues", [])) + "\n\n")
 
        # RECOMMENDATION / PRIORITY / SUPPORTING DOCUMENTS / EXPECTED IMPACT / RISK
        prompt = (
            "You are a strategy adviser to the Lufthansa Executive Board.\n"
            "Answer the goal using ONLY the evidence below. Do not invent facts or figures. "
            "You MAY draw reasonable strategic conclusions from the evidence. Write "
            "'INSUFFICIENT EVIDENCE' ONLY if NONE of the evidence below relates to the goal "
            "at all; if any evidence touches the topic, you MUST give a concrete recommendation.\n\n"
            "Goal: " + goal + "\n\n"
            + sentiment_note + "\n"
            + evidence_block(evidence) + "\n\n"
            + feedback +
            "Reply using exactly this structure:\n"
            "RECOMMENDATION: <a specific, actionable decision>\n"
            "PRIORITY: <High/Medium/Low>\n"
            "SUPPORTING DOCUMENTS: <the evidence numbers you rely on, e.g. 1, 3, 4>\n"
            "EXPECTED IMPACT: <benefit for the company>\n"
            "RISK: <financial/operational/strategic risk>"
        )
        rec = ask(prompt)
        revisions = state.get("revisions", 0) + (1 if feedback else 0)
        return {"recommendation": rec, "revisions": revisions}

    # Rule: Cited reference numbers must exist (1..N). Otherwise, mark as NEEDS_REVISION immediately
    def validate_node(state: AgentState):
        goal = state["goal"]
        evidence = state["evidence"]
        recommendation = state["recommendation"]
        n = len(evidence)
        valid_range = set(range(1, n + 1))
 
        # IDs are outside the permitted range – find out.
        out_of_range = []
        # Step 1: Retrieve all the IDs referenced from the recommendation
        cited = cited_ids(recommendation)
        # Step 2: Go through each of these IDs one by one.
        for c in cited:
            # Step 3: Check whether the ID is outside the valid range.
            if c not in valid_range:
                out_of_range.append(c)
 
        prompt = (
            "You are an INDEPENDENT validation module. Check the DRAFT recommendation "
            "strictly against the evidence. Be skeptical.\n"
            "Rules:\n"
            "1. Every factual claim must be supported by the evidence below.\n"
            "2. Cited evidence numbers must exist (valid range: 1.." + str(n) + ").\n"
            "3. Flag any invented facts, figures or events not present in the evidence.\n\n"
            "Goal: " + goal + "\n\n"
            + evidence_block(evidence) + "\n\n"
            "DRAFT RECOMMENDATION:\n" + recommendation + "\n\n"
            'Return ONLY JSON: {"verdict": "APPROVED" or "NEEDS_REVISION", '
            '"grounded": true or false, "issues": ["short issue", ...]}'
        )
       # LLM checker: checks, with a degree of scepticism, whether each statement is supported by evidence, and returns JSON {verdict, grounded, issues}
        parsed = extract_json(ask(prompt))
        if not isinstance(parsed, dict):
            parsed = {}
        verdict = parsed.get("verdict", "APPROVED")
        issues = parsed.get("issues", []) if isinstance(parsed.get("issues"), list) else []
        grounded = bool(parsed.get("grounded", True))
 
        # If necessary, the strict rule overrides the LLM (an ‘out_of_range’ error triggers ‘NEEDS_REVISION’)
        if out_of_range:   # hard rule overrides the LLM
            verdict = "NEEDS_REVISION"
            grounded = False
            issues = [f"Cited evidence {out_of_range} does not exist "
                      f"(only 1..{n} available)."] + issues

        # Refuse to give in: “INSUFFICIENT EVIDENCE” despite the existence of evidence
        # requires a correction to be made, rather than accepting the lack of a reply.
        if "INSUFFICIENT EVIDENCE" in recommendation.upper() and n > 0:
            verdict = "NEEDS_REVISION"
            grounded = False
            issues = [f"The draft refused with 'INSUFFICIENT EVIDENCE', but {n} relevant "
                      f"evidence items are available. Give a concrete, evidence-based "
                      f"recommendation citing real evidence numbers."] + issues
 
        return {"validation": {"verdict": verdict, "grounded": grounded,
                               "issues": issues, "revised": state.get("revisions", 0) > 0}}
    
    # If the verdict is NEEDS_REVISION and has not yet been revised (revisions < 1), it goes back to 'decide'. Otherwise, the graph ends (END)
    def needs_revision(state: AgentState):
        v = state.get("validation", {})
        if v.get("verdict") == "NEEDS_REVISION" and state.get("revisions", 0) < MAX_REVISIONS:
            return "decide"
        return END
    
    # Create a graph
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("decide", decide_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "decide")
    graph.add_edge("decide", "validate")
    graph.add_conditional_edges("validate", needs_revision, {"decide": "decide", END: END})
    return graph.compile()