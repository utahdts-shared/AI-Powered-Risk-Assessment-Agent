# Agent Architecture

What happens after a user clicks **Run Risk Analysis**.

The backend is a Google ADK 2.0 dynamic workflow. A single orchestrator node,
`risk_assessment`, drives the run in ordinary Python control flow, invoking each
child node via `ctx.run_node()`.

The graph is **static**: it is the same shape whether the risk matrix has three
rows or fifty. Everything that varies per assessment — the categories, their
level descriptions, the project details, the evidence — lives in session state,
and the orchestrator loops over it. There is one assessment agent, called once
per category, not one agent per category.

Only two node types call a model: `extract_url` and `assess_category`.
Everything else is deterministic Python, which keeps a run affordable on a
free-tier key and keeps the parts that must be reliable out of the model's hands.

```mermaid
flowchart TD
    START([User clicks Run Risk Analysis])

    subgraph INGEST["Phase 1 · Ingest — deterministic"]
        LOAD["<b>matrix_loader</b><br/>Parses the risk matrix within strict size,<br/>coordinate and shape limits."]
        SEED["<b>build_initial_state</b><br/>Turns the matrix and the user's answers into<br/>the session state the graph runs from:<br/>the category work list, levels and definitions."]
    end

    subgraph EVIDENCE["Phase 2 · Evidence — once per source, not once per category"]
        URL["<b>extract_url</b> · LLM<br/>Reads each URL with Gemini's url_context tool.<br/>Accepts the page only if the API confirms retrieval;<br/>otherwise the source is excluded."]
        PDF["<b>extract_pdf</b> · no LLM<br/>Extracts embedded text with pypdf, under page<br/>and time bounds. Flags scanned or image-only<br/>files, which yield no text — there is no OCR."]
        BRIEF["<b>evidence_brief</b><br/>Folds every usable extract into one bounded brief,<br/>and lists the sources that failed so the model<br/>treats them as absent."]
    end

    subgraph ASSESS["Phase 3 · Assessment — one agent, looped over the work list"]
        PICK["<b>risk_assessment</b> (the loop)<br/>Writes the next category from state into<br/><code>current_category</code>, then calls the agent."]
        AGENT["<b>assess_category</b> · LLM<br/>Rates whichever category state names.<br/>Its instruction is a callable that reads<br/>that category and its level definitions."]
        STORE["<b>finding_&lt;category&gt;</b><br/>The loop copies each result to its own<br/>state key, so per-category results stay<br/>individually inspectable."]
    end

    subgraph REPORT["Phase 4 · Report — deterministic"]
        COMPILE["<b>report.sanitize_result</b><br/>Forces each risk level back onto the matrix's<br/>own levels, and neutralises markup in the<br/>model's free text."]
        VIEW["<b>Results view</b><br/>One expander per category, colour-coded,<br/>plus any sources that could not be read."]
        XLSX["<b>report.build_report_workbook</b><br/>Writes the .xlsx with every cell stored as<br/>inert text, so no value can execute as a formula."]
    end

    START --> LOAD --> SEED
    SEED --> URL
    SEED --> PDF
    URL --> BRIEF
    PDF --> BRIEF

    BRIEF --> PICK
    PICK --> AGENT
    AGENT --> STORE
    STORE -->|"more categories in state"| PICK
    STORE -->|"work list empty"| COMPILE

    COMPILE --> VIEW
    COMPILE --> XLSX

    classDef llm fill:#fde68a,stroke:#b45309,color:#1c1917
    classDef det fill:#bfdbfe,stroke:#1d4ed8,color:#1c1917
    classDef out fill:#bbf7d0,stroke:#15803d,color:#1c1917
    class URL,AGENT llm
    class LOAD,SEED,PDF,BRIEF,PICK,STORE,COMPILE det
    class VIEW,XLSX out
```

Yellow nodes call a model. Blue nodes are deterministic Python. Green nodes are
what the user receives.

## The agents

| Agent | Type | What it does |
|---|---|---|
| `risk_assessment` | Orchestrator | Drives the whole run. Reads the category work list from session state, paces every model call against the free-tier quota, and emits the progress the UI shows. |
| `extract_url` | Function node, calls a model | Reads one URL. Accepts the content only when the API confirms the page was actually retrieved; a page that could not be read is excluded rather than answered from the model's prior knowledge. |
| `extract_pdf` | Function node, no model | Reads one PDF's embedded text locally, so it costs no quota. Reports files that yield too little text to be useful. |
| `assess_category` | LLM agent | Rates one risk category against the evidence brief and the project details. Returns a structured result whose risk level is constrained to the matrix's own levels. |

## Why one agent instead of one per category

`assess_category`'s instruction is a callable rather than a fixed string. ADK
invokes it before each request with a read-only view of session state, so the
loop can set `current_category` and the same agent rates whatever it finds
there. A custom matrix with different categories needs no change to the graph.

This is also what lets `agent.py` expose a module-level `root_agent` for
`adk run` and `adk web` to discover. A graph generated per matrix could not be
one, because it would not exist until a matrix had been uploaded.

Per-category results are not lost to the sharing: the agent has a single
`output_key` that each iteration overwrites, and the loop copies every result
into its own `finding_<category>` state key.

## Why the phases are split

Each assessment reads the **evidence brief**, never the raw sources. With nine
categories and up to ten sources, re-reading per category would mean up to
ninety retrieval calls per run. Extracting once holds a run to roughly
`URLs + categories` model calls — about eleven for the shipped matrix — which is
what makes it viable on a free-tier key.

## Failure handling

Each node catches its own errors, because a node that raises aborts the whole
ADK run. A category that fails becomes a single error row and the remaining
categories still run.

The exception is the free-tier **daily** quota. That does not clear while a run
is in progress, so the run stops, says so plainly, and marks the remaining
categories "Not assessed" rather than producing an identical error for each one.

## How the report is produced

Nothing the model returns reaches the user unchecked. Every finding passes
through `report.sanitize_result`, which forces the risk level back onto the
matrix's own levels — anything else becomes `Unknown` — and neutralises markup
in the free-text fields so they cannot render as links or images.

The workbook is then written with every cell stored as inert text, so a value
beginning with `=` is never evaluated as a formula by the spreadsheet client.
Both controls are load-bearing: the structured output schema constrains only the
risk level, and the reasoning and mitigation fields remain free text that a
prompt injection in a matrix cell, a vendor page, or a PDF can still influence.
