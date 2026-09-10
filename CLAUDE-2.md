# CLAUDE.md

Single operational reference for co-authoring Computer Science
conference, workshop, and journal articles in this folder. Covers the
working agreement, prose and structure rules, LaTeX and venue
mechanics, and the author's own style profile.

**Precedence.** Rules are ordered. When two conflict, the earlier wins:

1. **Section 1 — orienting-professor preferences.** Overrides
   everything else in this file. Argue against a rule here only with a
   concrete case (e.g., the venue template forbids the structure), and
   flag the conflict transparently.
2. **Sections 2–10 — the writing guide.** Fill in detail where
   Section 1 is silent.
3. **Part C — author style profile.** Descriptive, not normative. It
   never overrides a rule above it; where the guide is silent, follow
   it.

Parts A and B are procedural, not stylistic, and sit outside this
ordering.

This document is opinionated and prescriptive.

---

## Current work

| | |
|---|---|
| Venue / template | Springer LNCS (`llncs`) — configuration in §B.5 |
| Page limit | 12 pages |
| Citation style | Numbered, `splncs04.bst` (§6.1) |
| Density bracket | ~400–480 words/page (§7.2), i.e. roughly 4,800–5,800 words |

**Twelve pages in LNCS is a full paper, not a short one.** Two
consequences worth stating up front:

- Sections 1–10 apply unchanged. The structural omissions catalogued in
  §C.19 — no Related Work section, no contributions list, no figure, no
  research questions, no repository link, no confidence intervals — were
  defensible under six-page pressure and are not defensible here. Treat
  each as required unless the author rules otherwise.
- §C.17 becomes live. Run-in bold micro-headings (§C.5) were adopted as
  a short-paper default; at this length numbered subsections carry the
  §1.7 preambles and cross-references better. Ask before defaulting
  either way.

Update this block when the target changes. It is the only
venue-specific state in this file.

---

## Contents

- [Current work](#current-work)
- [Part A — Working agreement](#part-a--working-agreement)
- [1. Orienting-professor preferences (overrides all)](#1-orienting-professor-preferences-overrides-all)
- [2. Audience and framing](#2-audience-and-framing)
- [3. Prose economy](#3-prose-economy)
- [4. Section deep-dives](#4-section-deep-dives)
- [5. Tables, figures, algorithms](#5-tables-figures-algorithms)
- [6. Citations and references](#6-citations-and-references)
- [7. Paper-type templates and page budgets](#7-paper-type-templates-and-page-budgets)
- [8. Revision protocol](#8-revision-protocol)
- [9. Thesis-specific notes](#9-thesis-specific-notes)
- [10. Authoritative sources](#10-authoritative-sources)
- [Part B — LaTeX, build, and venue mechanics](#part-b--latex-build-and-venue-mechanics)
- [Part C — Author style profile](#part-c--author-style-profile)
- [Appendix: adding new orienting-professor feedback](#appendix-adding-new-orienting-professor-feedback)

---

# Part A — Working agreement

## A.1 Role and boundaries

- **Writing partner**, not a ghostwriter. Ask when intent is ambiguous.
- **Never invent** experimental results, data points, benchmark numbers,
  or citations. Flag missing items with `[CITE?]`, `[DATA?]`,
  `[TODO: …]`, `[DECISION NEEDED: …]`.
- **Never fabricate** related work. If asked to discuss a paper you
  don't know, say so.
- **Preserve the author's voice and ideas.** Improve clarity and flow
  without overwriting style. Part C catalogues which habits are voice
  and which are defects; consult it before rewriting prose. Explain
  substantial rewrites in 1–2 sentences.
- Write in **academic English**: precise, concise, active voice.
  Present tense for established facts; past for specific experiments.
  *"We"* even for single-author papers (CS convention).
- **Disclose AI assistance** — most venues require it. AI cannot be
  listed as an author. ACM/IEEE/CBSoft disclosure templates are in
  Section 10.

## A.2 Working mode

1. Ask for venue / template / page limit.
2. Ask for the one-sentence core contribution.
3. Propose an outline; get approval before drafting.
4. Iterate **section by section**: propose the prose for a section,
   wait for the author's go, then commit it to the LaTeX source.
5. After each section commit, rebuild and verify (page count, no
   overfull boxes, all citations resolve).

This assistant handles both prose drafting and LaTeX/citation
management.

## A.3 Interaction style

- When given a section to write, **ask what content/ideas to include**
  before drafting if not specified.
- When rewriting, **explain non-trivial changes** in 1–2 sentences.
- Suggest improvements but **present them as suggestions with
  reasoning** — don't force changes.
- Use `[CITE?]`, `[DATA?]`, `[TODO: …]`, `[DECISION NEEDED: …]`
  markers for gaps.
- When in doubt about a stylistic choice, **ask** rather than guess.
- Keep meta-discussion separate from draft content.
- Default to terse responses; the author reads diffs and PDFs directly.
- Commit only when asked — never on your own.

## A.4 Hand-off documents between sessions

Long-running work uses `handoff_*.md` files inside the article folder
to consolidate decisions, content references, and state between
sessions. These age fast — date them and rewrite rather than appending
forever. Examples: `articles/blockchain-principles/handoff_section_3_ledger.md`,
`articles/cardio-forense/handoff.md`.

---

# 1. Orienting-professor preferences (overrides all)

These rules take priority over every other section. Argue against them
only with a concrete, well-reasoned case (e.g., venue template forbids
the structure). Flag conflicts transparently.

## 1.1 Section order

```
1. Introduction
2. Related Work     (BEFORE the method, not after)
3. Proposal         (method; the centerpiece)
4. Experiments      (or Evidence, for ideas papers)
5. Conclusion
References
```

## 1.2 Introduction: five-fact framing

The Introduction walks through five facts in order:

1. **Scenario.** Broad context, the area of work.
2. **Challenges.** Where the scenario gets hard.
3. **Gap.** What the challenges are and why they remain unresolved.
   Compact literature snippet (2–4 sentences, 3–6 citations), not a
   second Related Work.
4. **Approach in principle.** How the problem could be solved.
5. **Proposal.** The paper's actual contribution.

Followed by a one-sentence section-by-section outline. The phrasing
**"The paper is divided as follows:"** is acceptable and overrides the
generic anti-pattern about TOC-paragraphs.

## 1.3 Related Work

- Organize by theme, not chronology.
- Most citations from the last ~3 years; older only for seminal works.
- Avoid single-source / single-school coverage. If the cited set comes
  from one author group, broaden the search.
- Compare; don't list. For each body of work: summarize → compare to
  the proposal → flag remaining gaps. No "Author X did Y" enumeration.
- Master's papers can survey ad hoc; PhD-level work should use a
  systematic review method and say so.
- The **final sentence of the final paragraph** must explicitly
  contrast the reviewed work with what the paper will do.

## 1.4 Proposal (method)

The Proposal is the paper's centerpiece. Draft it first; other sections
align to it.

- **Technology-independent.** The method must be reproducible with
  different tools or implementations. Tool-specific details belong in
  Experiments.
- **Reproducibility-first.** Include whichever of these is load-bearing:
  data structures, model definitions, pseudocode, architecture diagrams,
  step-by-step procedures.
- **Decision rationale is a first-class subsection** (not a footnote):
  the design decisions, alternatives considered, why the chosen approach
  was chosen.

## 1.5 Experiments

In roughly this order:

1. **Metrics.** Which metrics, stated explicitly. Never leave a metric
   implicit (e.g., "accuracy" as an unstated default).
2. **Metric definitions.** How each is computed.
3. **Experimental scenario.** What is being tested, under what
   conditions.
4. **Implementation details.** Technology, versions, public repository
   link.
5. **Data description.** Sources, sizes, splits, preprocessing.
6. **Experiments.** Each must correspond to something promised in the
   Introduction's five facts. If Intro promises X and Experiments tests
   Y, one of them is wrong.

Report confidence intervals alongside statistical test p-values when
making comparisons. State the interval method (bootstrap,
normal-approximation, Wilcoxon).

## 1.6 Conclusion

Three-beat rhythm:

1. **"The paper represented…"** — summary of what was done.
2. **"The experiments demonstrate…"** — what the experiments showed.
3. **"Therefore one can see…"** — synthesis: what this means for the
   field or for practice.

Followed by a Future Work paragraph that names limitations, points at
where the work could improve, and **explains how those improvements
could be achieved.** Open questions should come with at least a sketch
of the research path, not a wish list.

## 1.7 Section preambles

Every section with subsections opens with a one- or two-sentence
preamble that names the subsections and says **what each contributes
substantively**, not just labels them.

Good: *"Section 3.1 motivates X; Section 3.2 introduces Y."*
Weak: *"Section 3.1 covers preliminaries; Section 3.2 covers methods."*

## 1.8 Self-contained citations

Every claim that cites a reference must stand on its own without the
reader having to consult the reference. The citation supports the claim;
it does not replace the explanation. If a sentence reads coherently
only after the reader chases the citation, the sentence needs more
substance.

## 1.9 Paragraph and section quality gate

- Every paragraph: 3–4 sentences of substantive content.
- Every section: 3–4 paragraphs of substantive content.
- Every subsection: same.

These are quality gates, not structural mandates. A candidate paragraph
with one or two legitimate sentences must merge with an adjacent
paragraph or become a closing clause; a candidate section with two
legitimate paragraphs must fold into a related section. Do not pad.

## 1.10 Reproducibility and discoverability

A maintained public repository (code, data or pointers, scripts, env
spec, README) is looked on very favourably. Include the link in the
paper. The Proposal must be reproducible from the paper alone; the
experiments must be re-runnable from the repository.

## 1.11 Reading habit

Read five papers from a target venue before submitting. Calibrate tone,
format, depth. Hunt for citation gaps — they signal positioning
opportunities.

---

# 2. Audience and framing

This chapter governs how the paper treats its reader and how it
positions itself against prior catalogs, frameworks, or methodologies.
Most "the writing isn't careful with the reader" feedback resolves to
violations of this chapter.

## 2.1 Anchor every concept at first use

Identify the audience explicitly during planning: *who is the reader?*
A venue like SBES IIER reaches general software engineers, not subdomain
specialists. A NeurIPS paper reaches ML researchers but rarely
domain-specific applied researchers.

For every term that is not part of the audience's working vocabulary,
**anchor at first use**: a one-clause definition embedded in the
sentence that introduces it. Do not assume the reader will infer.

- Bad: *"The system uses MPC across federated participants to compute
  the aggregate without leaking inputs."*
- Good: *"The system uses secure multi-party computation (MPC) — a
  protocol class that lets several participants jointly compute a
  function over their inputs without revealing those inputs to each
  other — across the federated participants."*

Anchor once, then use the term freely. Anchor again only if the term
returns after a long absence (e.g., several sections later) and the
reader may have lost context.

Anchoring is not "dumbing down." A specialist reader skims an anchor in
half a second; a non-specialist reader cannot recover from an unanchored
introduction. The cost is asymmetric.

## 2.2 Do not conflate catalogs at different abstraction levels

When invoking prior catalogs (SOLID, GoF design patterns, Clean Code,
ACID, REST, twelve-factor app, etc.), be precise about the level each
operates at. They are not interchangeable.

| Level | Examples | What it is |
|---|---|---|
| Principle | SOLID, ACID, twelve-factor | Small, named, normative statements |
| Pattern | GoF, PoEAA, smart-contract patterns | Concrete, problem-specific solutions |
| Practice | Clean Code, Pragmatic Programmer | Habits and craftsmanship rules |
| Survey | ACM Computing Surveys reviews | Comprehensive maps of a subfield |

If the paper proposes an artifact at one of these levels, it sits next
to its peers at that level — not next to artifacts at the level above
or below. Bucketing them in a single sentence ("X, Y, and Z" where X,
Y, Z operate at different abstraction levels) is a precision error.
Name the level of each artifact explicitly when introducing it, and use
the prior catalog the paper is closest to (in both medium and
abstraction level) as the positioning anchor — not the broader bundle.

## 2.3 Lineage, not contrast, when proposing a peer artifact

When the paper proposes a new artifact (catalog, framework, model) that
is meant to stand at the same abstraction level as a well-known prior
artifact in a different domain, frame the relationship as **lineage or
parallel**, not as displacement.

Avoid: *displaces, inverts, abolishes, breaks, fails under, rethinks,
overturns, replaces.*

Prefer: *the analog of X for domain D, in the tradition of Y, sits
alongside, adopts the vocabulary level of, plays the role of X in this
medium.*

If a genuine contrast exists in a specific dimension, state it as a
parallel comparison — "X assumes a property that this medium does not
provide" — not as a takedown of X. The reader should leave with the
impression that both artifacts are valid; the new one extends the
catalog tradition into a new medium.

This is a vocabulary discipline. The structural content (which prior
principle the new one is closest to, where the assumption stops
holding) can stay; the rhetorical posture must shift.

## 2.4 Justify "why this many, why these"

A paper proposing a catalog with a specific cardinality (six principles,
seven heuristics, twelve factors) must justify the cardinality
explicitly. Defaults that fail:

- "We propose six because the acronym spells a word." (Mnemonic, not
  argument.)
- "We propose six and leave others for future work." (Punt.)
- "The literature suggests these are important." (Cite which work,
  why these and not their neighbours.)

Defensible justifications:

- **Derivation from structural properties.** "The medium has N
  irreducible structural properties; each generates one principle."
  Anchor the N against an independent systematization in the
  literature.
- **Coverage of failure modes.** "These N principles together address
  the recurring failure classes documented in the field." Anchor on a
  taxonomy of failures.
- **Hybrid.** Properties yield the catalog; failure analysis validates
  it. Strongest argument.

Always include a "What we leave out" passage that names plausible
candidates (privacy, scalability, consensus-aware design, etc.) and
either folds them into an existing principle, marks them as quality
attributes one level below the catalog, or earmarks them for a future
revision. Honest exclusion is more credible than a closed set.

## 2.5 Reader-experience checks (apply during revision)

For each section, answer aloud:

- Can a reader from the target venue follow this section after one
  read?
- Is every domain term anchored at first use?
- Does any sentence assume context the reader does not have?
- Does any paragraph state a conclusion without showing the reasoning?

If any answer is no, revise before moving on.

---

# 3. Prose economy

This chapter consolidates paragraph- and sentence-level rules. Apply to
every section, **including the abstract** (abstracts are the most
visible escape route for filler prose; do not exempt them).

## 3.1 Paragraph structure (M.E.A.L.)

Every body paragraph:

- **M**ain point: topic sentence first; one claim per paragraph.
- **E**vidence: data, citations, examples, or formal results.
- **A**nalysis: what the evidence means.
- **L**ink: transition to the next paragraph.

Topic-sentence test: reading only the first sentence of every paragraph
in a section should yield a coherent summary of the section's argument.
If it does not, the topic sentences are weak.

## 3.2 Known-new contract

Each sentence opens with what the reader already knows (from prior
sentences or general knowledge) and ends with new information. New
material in sentence N becomes the known material in sentence N+1.

Violating this contract forces the reader to re-read.

## 3.3 Paragraph length

- Target 4–8 sentences.
- Minimum 3 sentences (per §1.9 quality gate).
- Maximum 10. Longer paragraphs usually contain two ideas.
- Transitional paragraphs may be 2–3 sentences.

## 3.4 Voice and tense

- Active voice by default. *"We propose…"* not *"A method is proposed…"*
- Passive only when the actor is irrelevant.
- Present tense for established facts; past tense for specific
  actions in the paper's experiments.
- "We" even in single-author papers (CS convention).

See also §C.6, which records the author's established tense split.

## 3.5 The lero-lero detector (filler prose to cut)

The phrases below are filler. Cut on sight; do not negotiate.

| Cut | Replacement |
|---|---|
| It is interesting to note that… | (state the fact) |
| It should be noted that… | (state the fact) |
| It is clear that… | (state the claim) |
| Needless to say… | (then don't say it) |
| In order to… | To… |
| Due to the fact that… | Because… |
| A large number of… | Many… |
| In the case of… | For… / When… |
| Has the ability to… | Can… |
| At the present time… | Currently… |
| In light of… | Given… |
| With respect to… | For… / On… |
| It is worth mentioning that… | (cut) |
| As a matter of fact… | (cut) |

## 3.6 Adjective and clause stacks

Stacking adjectives or relative clauses signals throat-clearing, not
content.

- Stop at two adjectives per noun. *"A small, named catalog"* is fine;
  *"a small, named, contrast-anchored catalog"* is filler.
- Stop at one relative clause per sentence. Chained *"through which…
  through which…"* clauses signal a sentence that should be split.
- One emphatic per claim. *"Deliberately specific abstraction level"*
  → *"specific abstraction level"*. *"Deliberately narrow move"* →
  *"narrow move"*. The intensifier is doing work the verb should do.

## 3.7 Em-dashes

Em-dashes (`---` in LaTeX, `—` in Unicode) are correct punctuation but
encourage stacking ideas in a single sentence. Default to commas,
parentheses, colons, or sentence breaks.

- *`a — b — c`* lists → serial-comma list: *"a, b, and c."*
- *`clause — elaboration — clause`* → parentheses or two sentences.
- Reserve the em-dash for emphatic breaks no other punctuation handles.

A paper with more than ~3 em-dashes per page is almost certainly
overusing them. Em-dash-heavy prose also reads as AI-generated, which
is an increasingly costly secondary signal.

## 3.8 Precision and specificity

- Bad: *"The results were good."*
- Good: *"F1 improved by 4.2 percentage points."*
- Bad: *"We used a large dataset."*
- Good: *"We used ImageNet-1K (1.28M training images, 1000 classes)."*

See §C.13 for the recurring version of this defect in the author's
drafts.

## 3.9 Hedging discipline

- Strong claim (proof / overwhelming evidence): *"X holds when…"*
- Standard claim (experiments): *"Our results indicate…"*
- Tentative claim (speculation): *"One possible explanation is…"*

Never hedge with strong evidence. Never assert certainty without it.

## 3.10 Terminology consistency

Pick one term per concept and use it throughout. Maintain a small
terminology table during planning; do not let synonyms drift in mid-paper.

## 3.11 CS-specific language fixes

- *Performant* — not a standard word. Use *high-performing* or *efficient*.
- *Leverage* — use *use* or *exploit*.
- *Utilize* — use *use*.
- *Aforementioned* — use *this* or *the above*.
- *Respectively* — avoid; it forces the reader to count items.
- Numbers: spell out one through nine; digits for 10 and above, all
  measurements, all experimental results.
- *Data* is plural in formal writing (*"the data show…"*); *"data is"*
  is accepted in modern CS — pick one and stay consistent.
- Hyphenate before a noun (*"well-known method"*), not after a verb
  (*"the method is well known"*).

## 3.12 Register leaks (cut on sight)

These read as informal, salesy, or AI-generated. Detectable patterns:

| Leak | Example | Fix |
|---|---|---|
| Slogan / antithesis as a claim | "competitive, not dominant"; "fast, not free" | State the claim with the quantity. |
| Planning metaphor in prose | "rung by rung", "the summit", "climbing the staircase" | Describe the actual logical step. |
| Cutesy compound modifier | "single-digit-megabyte memory"; "byte-level certificate" | Give the number or the mechanism (9.3 MB; bit-identical, directly verifiable). |
| Punchline fragment | a sentence fragment for emphasis | Write a full sentence. |

Planning scaffolds (staircase, rungs, corners-as-slogans) belong in
notes and this file, never in the paper.

**Open conflict:** the author's habit of personifying models and results
(§C.16) falls near this rule and is unresolved. Do not silently apply
§3.12 to it.

---

# 4. Section deep-dives

This chapter expands Section 1 where it is silent. When a rule here
conflicts with Section 1, Section 1 wins.

## 4.1 Title

Formula: *[Method/what you do] + [for/on/via] + [Problem/Domain]*.

Avoid:
- *"A Novel Approach to…"* — *novel* is the reviewer's call.
- *"Towards Better…"* — vague.
- *"An Investigation of…"* — passive.
- *"Some Results on…"* — undersells.
- Puns that obscure content (unless the pun *is* the artifact's name).

Test: can a researcher scanning proceedings tell, from the title alone,
whether this paper is relevant?

## 4.2 Abstract

Template:

```
[MOTIVATION: 1–2 sentences on why the area matters]
[PROBLEM:    1 sentence on the specific gap]
[APPROACH:   1–2 sentences on the method (not details)]
[RESULTS:    1–2 sentences with concrete numbers]
[IMPACT:     1 sentence on broader significance]
```

Rules:
- No citations; no figure/table/equation references; no undefined
  acronyms.
- Do not open with *"In this paper, we…"*.
- Quantitative results, not *"good results."*
- Independently understandable — a reader of only the abstract should
  grasp problem, approach, and result.
- **Abstracts obey Section 3 (prose economy) in full.** Most abstract
  prose problems come from authors treating the abstract as a separate
  register where adjective stacks and filler are tolerated. They are
  not.

## 4.3 Introduction

Follow §1.2 (five-fact framing) and Section 2 (audience, framing).

Final paragraph(s):

- Contributions list of 2–4 specific, refutable claims. Each item must
  be testable: a reviewer can write *"yes, they did this"* or *"no,
  they didn't."* Vague items ("We explore…", "We provide insights into…")
  are not contributions.
- One-sentence section outline. *"The paper is divided as follows:"*
  is acceptable per §1.2.

## 4.4 Related Work

Follow §1.3. Template per thematic group (1–2 paragraphs):

```
[Topic sentence: what this group does]
[Key papers + brief description of their approach]
[Strengths — be fair]
[Specific limitation the proposal addresses]
```

Close with 2–3 sentences positioning the proposal relative to the
reviewed work, using lineage vocabulary per §2.3.

## 4.5 Proposal (method)

Follow §1.4.

Opening: high-level overview with a figure (architecture, pipeline) and
1–2 paragraphs of intuition **before** any formalism. The reader
understands WHAT before HOW.

Per component:

1. Challenge addressed (1 sentence).
2. Intuition (1–2 sentences).
3. Formal description (equations, algorithm, design).
4. Justification — including alternatives considered.

Algorithms: pseudocode, not source code. Include Input/Output, line
numbers, complexity. Use `algorithm2e` or `algpseudocode`.

Do not include: debugging history, false starts, implementation quirks
that do not affect the method conceptually.

See §C.1: every display equation is followed by a prose sentence
explaining what it does.

## 4.6 Experiments (or Evidence)

Follow §1.5.

Opening: state the research questions explicitly.

```
We design experiments to answer:
- RQ1: Does X outperform existing approaches on T?
- RQ2: How does each component of X contribute?
- RQ3: How does X scale with parameter P?
```

For ideas papers and conceptual contributions, the equivalent section
may be titled **Evidence** rather than Experiments. Same rules apply:
state the questions, anchor the data, report results with numbers,
interpret rather than describe ("Method X achieved 94.3%, a 3.1-point
improvement over the strongest baseline" — not "Table 1 shows 94.3%").

Ablation study is mandatory for ML / systems papers with proposed
components.

## 4.7 Discussion

Often merged with Results in conference papers. When separate:

1. Summary of main findings (not repetition).
2. Interpretation: why did the method work? What drives the
   improvement? What fails?
3. Limitations: specific and honest.
   - Bad: *"Our method has some limitations."*
   - Good: *"Our method requires O(n²) memory, limiting applicability
     to graphs with fewer than 50K nodes."*
4. Broader implications and concrete future work directions.

## 4.8 Conclusion

Follow §1.6. Synthesis test: if the conclusion could be generated by
copy-pasting the introduction and changing tense, it fails. A good
conclusion adds an insight, a connection, or a perspective that wasn't
obvious from the individual sections.

---

# 5. Tables, figures, algorithms

## 5.1 Choosing the format

| Data | Format |
|---|---|
| Exact numerical comparisons | Table |
| Trends over time / parameters | Line plot |
| Distributions | Histogram or box plot |
| Proportions | Bar chart |
| Architecture / pipeline | Diagram (TikZ, draw.io) |
| Algorithm steps | Pseudocode |
| Qualitative results | Side-by-side images |

## 5.2 Tables

- Caption above.
- `booktabs`: `\toprule`, `\midrule`, `\bottomrule`. No vertical rules.
- Bold best in each column; underline second best if relevant.
- Units in column headers; align decimals.
- Include ± std dev for multi-run averages.
- Every table referenced and interpreted in text. Do not paste a table
  with no surrounding analysis. See §C.14 for the recurring failure
  mode: a heading that promises analysis the prose never delivers.

## 5.3 Figures

- Caption below.
- Caption is self-contained: what is shown AND the key takeaway.
  - Bad: *"Results of our experiments."*
  - Good: *"Accuracy vs. epochs on CIFAR-10. Our method (blue) converges
    2× faster and reaches 3.1% higher final accuracy than the baseline
    (red)."*
- Vector formats (PDF, SVG) only — never rasterised screenshots of
  diagrams.
- Colour-blind-friendly palette (Wong, viridis).
- Readable in grayscale: use line styles in addition to colours.
- Label axes with units. Font ≥ 8pt after scaling.

## 5.4 Algorithms

- `algorithm2e` or `algpseudocode`.
- `\KwIn`, `\KwOut`, `\KwResult` as applicable.
- Line numbers.
- Comments on non-obvious steps.
- Complexity stated in caption or following paragraph.
- Pseudocode is not source code; keep abstraction level uniform.

---

# 6. Citations and references

## 6.1 Style by venue

- IEEE: numbered brackets in order of first appearance.
- ACM (`acmart`): author-date or numbered per template variant.
- LNCS: numbered.

Match the template; do not mix.

## 6.2 BibTeX hygiene

- Key format: `authorYEARkeyword` (e.g., `vaswani2017attention`),
  lowercase.
- Always include `doi`.
- Full venue names in `booktitle`, not abbreviations.
- Verify entries against the actual publication.
- Cite the published version of arXiv preprints when one exists.
- Remove auto-generated fields you don't use (`abstract`, `file`).

## 6.3 Citation practice

- Cite to support a claim, not to demonstrate reading volume.
- Each citation serves a purpose: ground a technique, establish a
  baseline, credit an idea.
- For seminal works, use author names: *"Transformers
  (Vaswani et al., 2017)…"*
- Self-citations in double-blind submissions: third person. *"Prior
  work [N] showed…"* not *"In our prior work [N], we showed…"*

---

# 7. Paper-type templates and page budgets

## 7.1 Section structure by paper type

| Type | Structure |
|---|---|
| Generic conference paper (IEEE/ACM, 6–8pp) | Intro · Related · Method · Experiments · Conclusion |
| Ideas / position paper (~6–8pp) | Intro · Related · Proposal · Evidence · Discussion · Conclusion |
| Systems paper (SOSP, OSDI; 12–14pp) | Intro · Background · Design · Implementation · Evaluation · Related · Conclusion |
| ML paper (NeurIPS, ICML; 8pp + appendix) | Intro · Related · Method · Experiments · Conclusion · Appendices |
| Workshop / short paper (4pp) | Intro (compressed funnel) · Approach · Preliminary Results · Conclusion |
| Journal article | As conference paper + extended Related · deeper method · more experiments · separate Discussion (12–20pp) |

Master's and PhD thesis structures are in Section 9.

## 7.2 Template density

Page counts are not a universal unit. The same prose runs different
lengths in different templates.

| Template | Words/page | Notes |
|---|---|---|
| NeurIPS / ICML / ICLR | ~550–650 | Single-column, tight margins, 10pt |
| IEEE two-column | ~750–900 | Dense; compact bibliography |
| ACM `acmart` (sigconf) | ~700–850 | Similar to IEEE |
| Springer LNCS | ~400–480 | **Generous.** 20–30% fewer words/page than NeurIPS |
| Thesis | ~300–400 | Double-spaced; wide margins |

## 7.3 Float overhead

A half-page figure or table costs slightly more than its size (~1.1–1.3×)
due to caption + margins + `\textfloatsep`. Tables using `\resizebox` or
`\scriptsize` increase effective cost: captions and surrounding
whitespace do not shrink with the body.

## 7.4 Page-budget compression rule

LaTeX does not reduce page count linearly with prose cuts. When cuts
happen on pages with floats, previously-displaced floats migrate back
into the freed space. Assume 50–70% of prose cuts translate to page
savings when floats are involved; ~90% on pure-text pages.

Heuristic when planning a compression pass:

```
lines_to_cut = pages_to_save × lines_per_page × 1.4
```

The 1.4 factor absorbs float reflow. **Over-plan compressions by
1.3–1.5×** so margin exists for one render rather than two. A list of
optional cuts is cheap to prepare; a mid-compile shortfall forces
triage under time pressure.

## 7.5 Pre-compression checklist

Before committing to cuts, confirm:

- Current page count and target page count.
- Template density bracket.
- Float locations, especially `\resizebox`-using floats.
- Bibliography position: own page or shared.
- Whether appendix or bibliography cuts are available as fallback.

After any compression pass, run the §C.12 check: cuts are the main
source of orphaned forward references.

---

# 8. Revision protocol

## 8.1 Multi-pass strategy

| Pass | Read | Check |
|---|---|---|
| 1. Structure | Headings, topic sentences, contributions | Coherent story? Contributions match content? Section order? |
| 2. Argumentation | Full text, ignore style | Every claim supported? Logical gaps? Limitations addressed? |
| 3. Reader experience | Section by section | Audience anchored (§2.1)? Catalogs not conflated (§2.2)? Lineage framing (§2.3)? |
| 4. Clarity | Paragraph by paragraph | One idea per paragraph? Topic sentences as summary? Known-new contract? |
| 5. Sentences | Sentence by sentence | Active voice? Concise? Consistent terminology? Lero-lero (§3.5) removed? Adjective stacks (§3.6) cut? |
| 6. Author-specific | Sentence by sentence | Part C.2 checklist: comma splices, long colon sentences, connective monotony, orphaned references, vague magnitudes, undelivered analysis. |
| 7. Polish | Formatting | Figures/tables referenced? Citations complete? Page limit respected? |

## 8.2 The Figure 1 test

A colleague reading only the title, abstract, Figure 1, and
contributions list should:

1. Understand the problem.
2. Understand the approach at high level.
3. Know the main result.

If not, revise these elements until the test passes.

## 8.3 The read-aloud test

Read the paper aloud (or use TTS). Awkward phrasing, run-on sentences,
missing transitions become obvious when heard. The lero-lero detector
(§3.5) catches more on a read-aloud pass than on silent reading, and so
do the comma splices in §C.9.

## 8.4 Common mistakes catalog

| Mistake | Detection | Fix |
|---|---|---|
| No clear contribution by page 3 | Reader can't state contribution after Intro | Add explicit contributions list earlier |
| Related Work is a list, not a review | Consecutive *"X did A. Y did B."* | Reorganize thematically; add comparative analysis |
| Method jumps to formalism | First equation appears with no intuition | Add overview paragraph + figure before formalism |
| No ablation study | Reviewer asks "Is component X necessary?" | Add table: full model vs. each ablated variant |
| Evaluation doesn't match contributions | Contribution X, evaluation measures Y | Realign: each contribution gets corresponding evidence |
| Vague claims | *"Our method is faster"* | *"Reduces inference time by 40% (120ms → 72ms)"* |
| Overclaiming | *"We prove NNs solve NP-hard problems"* | Soften: *"…finds near-optimal solutions in practice"* |
| Tense inconsistency | Mixed tense across method/experiments | Pick a tense per context |
| Dangling modifier | *"Using gradient descent, the loss decreased"* | *"Using gradient descent, we decreased the loss"* |
| Undefined acronym | *"We evaluate on OGB"* | *"…the Open Graph Benchmark (OGB)"* on first use |
| Pronoun ambiguity | *"It outperforms it on this dataset"* | Name subjects: *"GraphPool outperforms GCN-v2 on Cora"* |
| Caption says only "Results" | Pass §5.3 caption test | Self-contained caption with key takeaway |

---

# 9. Thesis-specific notes

## 9.1 Thesis statement (vs. paper contributions)

A thesis requires a single overarching argument the committee can
agree or disagree with — 1–2 sentences. Example:

> *"This thesis demonstrates that hierarchical graph pooling, when
> combined with structure-preserving objectives, enables scalable graph
> classification without sacrificing subgraph-level discriminative
> information."*

A paper has a contributions list; a thesis has a thesis statement plus
chapter-level contributions that support it.

## 9.2 Literature Review (NOT Related Work)

A thesis literature review is different from a paper's Related Work:

- **Scope**: the entire relevant field, not only competing methods.
- **Foundational concepts**: define key terms / theories. Textbook-level
  material is acceptable for committee members outside the exact
  subfield.
- **Thematic reviews**: per-theme survey + critical analysis + remaining
  gaps.
- **Synthesis**: explicit research-gap statement at the end.
- **Length**: 15–25pp master's, 20–30pp PhD.

Critical difference: Related Work positions a contribution; Literature
Review demonstrates mastery and motivates research questions.

## 9.3 Paper-based (sandwich) thesis

When published papers become chapters:

- **Chapter intro** (1–2 paragraphs): connect to the thesis argument
  and to the previous chapter.
- **Chapter summary + bridge** (1–2 paragraphs): contribution to the
  thesis + preview of the next chapter.
- **Unified terminology**: if Paper 1 says *"graph pooling"* and Paper 2
  says *"graph coarsening,"* pick one. Footnote the equivalence.
- **Unified notation**: harmonize symbols.
- **Contributions matrix**: a table (in Intro or Conclusion) mapping
  each contribution to the chapter and evidence.

## 9.4 The defence framing

Write as if convincing a skeptical committee member that:

1. You understand the field deeply (Literature Review).
2. You identified a meaningful gap (Intro + Lit Review).
3. You developed a technically sound approach (Methodology).
4. You validated your claims rigorously (Evaluation).
5. The work advances the field (Conclusion).

---

# 10. Authoritative sources

Read at least three of these once; revisit before any major paper:

| Resource | Author | Focus |
|---|---|---|
| *How to Write a Great Research Paper* (talk) | Simon Peyton Jones | Overall paper strategy |
| *Tips for Writing Technical Papers* | Jennifer Widom (Stanford) | Introduction structure |
| *Writing Technical Articles* | Henning Schulzrinne (Columbia) | Systems papers, style |
| *How to Write a Paper* | Oded Goldreich (Weizmann) | Theory papers, reader-first |
| *Three Sins of Authors in CS and Math* | Jonathan Shewchuk (CMU) | Anti-patterns |
| *Writing for Computer Science* (book) | Justin Zobel | Comprehensive book-length |
| *Paper: Introduction / Methods / Discussion* | MIT EECS CommKit | Section-by-section guidance |
| *The Elements of Style* | Strunk & White | Foundational English prose |

## LaTeX templates

- IEEE: `IEEEtran` (ieee.org)
- ACM: `acmart` (acm.org)
- Springer LNCS: `llncs` (springer.com)
- NeurIPS / ICML: per-year templates

## AI-co-authoring disclosure

ACM:
> *During the preparation of this work, the authors used [tool] for
> [specific use]. The authors reviewed and edited all AI-assisted
> content and take full responsibility for the publication's content.*

IEEE (Acknowledgments):
> *The authors acknowledge the use of [tool] for [specific use]. All
> content was reviewed, verified, and approved by the authors.*

AI cannot be listed as an author.

---

# Part B — LaTeX, build, and venue mechanics

## B.1 Per-article folder layout

Each article lives in its own subfolder (e.g., `f4rms/`,
`blockchain-principles/`, `cardio-forense/`) with the structure:

```
<article>/
  main.tex            # symlink -> sections/main.tex
  sections/
    main.tex          # preamble + \input{} for each section
    intro.tex
    related.tex
    proposal.tex      # or ledger.tex / wisard.tex / evidencias.tex
    evidence.tex      # or results.tex / discussion.tex
    conclusion.tex
  figures/            # all .png/.pdf/.svg assets cited by the .tex files
  references.bib      # BibTeX bibliography
  Makefile            # tectonic-based build
  <class files>       # e.g. acmart.cls + .bbx/.cbx/.bst, or llncs.cls
  main.pdf            # build artifact (gitignored)
  CLAUDE.md           # article-specific instructions (venue, state, restrictions)
```

The root `main.tex` is a symlink to `sections/main.tex` so tectonic and
Overleaf both find a top-level main file while the preamble lives next
to the section files it `\input`s.

The per-article `CLAUDE.md` is **optional and reserved for
article-specific content**: anchor references, content restrictions,
state of in-progress drafts. Nothing in this file should be duplicated
there. While a single article is active, its venue and page limit live
in the *Current work* block at the top of this file rather than in a
per-article one; split them out only when two articles are in flight at
different venues.

## B.2 Build

```bash
make           # tectonic compile -> main.pdf
make open      # open main.pdf
make clean     # remove aux/log/synctex/etc.
make distclean # also remove main.pdf
make zip       # produce <main>.zip ready to upload to Overleaf
```

Tectonic auto-downloads missing TeX packages on first run; no local
TeX Live needed.

After every edit, rebuild with `make` (or `touch main.tex && tectonic
--synctex main.tex`) and verify the page count with
`pdfinfo main.pdf | grep Pages`. Inspect float-heavy pages with
`pdftotext -layout -f N -l N main.pdf - | head` to see where text and
floats actually land.

## B.3 Choosing the class

| Venue | Class | Source |
|-------|-------|--------|
| CBSoft (SBES, SBLP, SBCARS, SAST, workshops) | `acmart` (sigconf) | CBSoft ACM-like template zip |
| Springer LNCS proceedings **(current target, §B.5)** | `llncs` | Springer's `llncs.cls` + `splncs04.bst` |
| SBC (Brazilian Computer Society standalone) | plain `article` + `sbc-template.sty` | SBC template Overleaf project |
| IEEE conference | `IEEEtran` | ieee.org template |
| NeurIPS / ICML | per-year `.sty` from the venue | venue website |

When starting a new article, unzip the conference template into the
article folder, move the `.cls` / `.bst` / `.bbx` / `.cbx` / `.dbx`
files to the article root, then create `sections/` and the section
skeleton.

## B.4 CBSoft `acmart` configuration

The CBSoft template is `acmart` with the ACM-specific bits stripped:

```latex
\documentclass[sigconf,anonymous,review]{acmart}   % drop anonymous,review for camera-ready
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\setcopyright{none}
\acmConference[SBES 2026]{40th Brazilian Symposium on Software Engineering}{September 8--11, 2026}{S\~ao Paulo, SP, Brazil}
```

Pick the right `\acmConference` line for the track:

- **SBES** (Research, Education, IIER, Tools, Industry, **IDEIAS**): `[SBES 2026]{40th Brazilian Symposium on Software Engineering}`
- **SBLP**: `[SBLP 2026]{30th Brazilian Symposium on Programming Languages}`
- **SBCARS**: `[SBCARS 2026]{20th Brazilian Symposium on Software Components, Architectures, and Reuse}`
- **SAST**: `[SAST 2026]{11st Brazilian Symposium on Systematic and Automated Software Testing}`
- **Workshops**: `[<Acronym> 2026]{<Workshop Full Name>}`

Camera-ready checklist:

- Remove `anonymous,review` from `\documentclass`.
- Fill in real `\author{}` / `\affiliation{}` / `\email{}` blocks.
- Set `\renewcommand{\shortauthors}{Surname et al.}` and `\renewcommand{\shorttitle}{...}`.
- Section titles in **Title Case**.
- Figure captions **below**, table captions **above** the float.
- Mandatory `\section*{Artifact Availability}` (or `Disponibilidade de Artefatos` in PT) between Conclusion and Acknowledgements.
- For Portuguese: `\renewcommand\abstractname{Resumo}`, `\renewcommand\keywordsname{Palavras-chave}`, `\renewcommand\refname{Refer\^encias}`, `\section*{Agradecimentos}`.

## B.5 Springer LNCS configuration

The current target (see *Current work*). Class and bibliography style
ship together in Springer's author kit:

```latex
\documentclass[runningheads]{llncs}

\title{...}
\titlerunning{...}          % required when the title is long
\author{First Author\inst{1} \and Second Author\inst{2}}
\authorrunning{F. Author et al.}
\institute{PESC/COPPE/UFRJ, Rio de Janeiro, Brazil \\
  \email{...} \and ...}

...

\bibliographystyle{splncs04}
\bibliography{references}
```

Notes specific to this class:

- Numbered citations, ordered by first appearance. Do not mix with
  author-date.
- `\keywords{}` inside the abstract environment; Springer expects 3–5.
- Figure captions below, table captions above — same as `acmart`.
- `llncs` supplies its own `\section`/`\subsection` numbering; no
  artifact-availability section is mandated, so the repository link
  required by §1.10 goes in the Experiments implementation-details
  paragraph (§1.5, item 4).
- LNCS is the most generous bracket in §7.2. Prose that fits 12 LNCS
  pages will not fit 12 IEEE or `acmart` pages; do not reuse a page
  estimate across templates.

## B.6 Sectioning conventions

- One section per file under `sections/`, `\input`ed from `sections/main.tex`.
- Each section file starts with `\section{...}` and a `\label{sec:...}` so cross-references work across files.
- Keep figures in `figures/` and reference with `\includegraphics{figures/foo}`.
- Bibliography keys per §6.2.
- Every section with subsections opens with a preamble per §1.7.

## B.7 Working on a section

Whether the prose comes pre-drafted from the author or is written
collaboratively in this session:

1. Identify which section file it belongs to (intro / related / proposal / evidence / conclusion / etc.).
2. Convert prose to LaTeX: `\textbf{}` / `\textit{}` / `\emph{}`, `\cite{key}` for citations, `itemize` / `enumerate` for lists, math in `$...$` or `equation` envs, code in `verbatim` or `lstlisting`.
3. If new citations appear, add the BibTeX entry to `references.bib` per §6.2. If the author doesn't provide an entry, look it up before guessing (DOI resolution preferred over hand-construction).
4. Replace the prior content in the target section file. Don't append after the placeholder; overwrite it.
5. Run `make` to confirm the article still compiles. Surface any TeX errors immediately.
6. Note any cross-references (figures, tables, sections) the section introduces, so the author knows what other files need updates.

## B.8 Uploading to Overleaf

1. `make clean && make zip` produces `main.zip` excluding build
   artifacts and template sample assets.
2. Overleaf: **New Project > Upload Project > select `main.zip`**.
3. In Overleaf, **Menu > Compiler: XeLaTeX** (matches local tectonic
   output; pdfLaTeX renders acmart with different font metrics and
   slightly different page breaks).
4. **Menu > Main document: `main.tex`**.

To update an existing Overleaf project file-by-file (preserves
collaborators, comments, history), upload the changed `.tex` / `.bib`
files via the Upload button in the file tree. Place files in their
matching subfolder (e.g., `sections/evidence.tex` goes inside
`sections/`).

## B.9 Reference patterns

- `f4rms/` — working example of the LNCS variant (different class, same layout principles).
- `blockchain-principles/` — working example of the CBSoft acmart variant (SBES IIER track).
- `cardio-forense/` — working example of the SBC template variant (plain article + `sbc-template.sty`).

---

# Part C — Author style profile

Descriptive, not normative. **Does not override Sections 1–10.** Where
they are silent, follow this.

Two jobs. §C.1–C.8 tell a co-author which habits to *preserve* when
rewriting, so edits don't sand the voice off. §C.9–C.15 are a revision
checklist of recurring defects (pass 6 in §8.1). §C.16–C.18 are open
questions only the author can settle.

**Evidence base.** One paper: *KalWeNN: Kalman-Filtered Memory for
Weightless Neural Network Classifiers* (WFD 2026, SBC template, seven
authors). Confidence is **low-to-moderate**. Co-authored prose cannot
be attributed. Short-paper page pressure explains some patterns that
would otherwise read as preference. Revisit after two or three more
single-author or first-author drafts.

## C.1 Read every display equation back in words

The strongest habit in the sample. After a display equation, one
sentence in prose says what the equation *does*, usually by naming a
trade-off or a limiting behaviour rather than restating the symbols.

> *The Kalman gain balances prediction and observation: when the
> predicted covariance is large relative to R the observation
> dominates, while a confident estimate yields only small corrections.*

Variants in the sample: naming what the equation makes visible
(*"Equation (7) makes the contrast with the classical scheme
explicit"*), and stating limit cases (*"For Q → 0, the update converges
to the empirical activation frequency"*).

**Rule:** never leave a display equation as the last element of a
paragraph. If a rewrite drops the read-back sentence, that is a
regression, not a compression.

## C.2 Mechanism, not just measurement

Empirical claims are followed by a causal account of *why* the number
came out that way. This is the clearest quality signal in the sample
and the thing most easily lost in editing.

> *The run-to-run randomness […] flips whole recognitions in discrete
> steps in the classic model, whereas in the Kalman model it only
> nudges continuous values that average out once summed.*

**Rule:** every results paragraph earns at least one mechanism
sentence. Reporting without explanation is a draft, not a section.

## C.3 Calibrated comparatives, and honesty about small gains

The default claim verb is *matches or improves*, not *outperforms*.
Datasets where the method wins by nothing are named, with the reason:
*"on the easy ones […] the gain is small, since every model already
classifies almost everything correctly."* Same for parameters:
*"within the tested grid these shifts are marginal."*

**Rule:** do not upgrade a hedge to a stronger claim during editing. Do
not delete a sentence that concedes a null result to save space — cut
elsewhere. Consistent with §3.9, and preserved as voice rather than
merely tolerated.

## C.4 Contrast anchored to the concrete prior mechanism

Novelty is framed against a specific named mechanism (*the unbounded
integer counter*), never against vague prior art. The comparison object
is always something a reader can picture. When positioning, name the
artefact being replaced at the level of its mechanism, not its
literature label.

## C.5 Run-in bold micro-headings

The sample uses bolded lead-in phrases (*Input binarization.*,
*Experimental setup.*, *Stability and Computational Cost Results.*)
instead of numbered subsections. Sections 1–10 are silent on this;
treat it as the default for short papers. **Not settled for the current
12-page LNCS target** — see §C.17.

Two fixes needed. Capitalization is inconsistent in the sample (mixed
sentence case and Title Case) — **pick sentence case and hold it**. And
per §1.7, a section carrying run-in headings still needs its opening
preamble naming what each contributes; Section 4 of the sample skips
it.

## C.6 Tense split

Past for procedure (*"All experiments were run on…"*, *"We evaluated
the models on…"*), present for findings (*"the Kalman update matches or
improves…"*, *"Table 1 brings the results together"*). Consistent in
the sample; keep as an explicit rule refining §3.4.

## C.7 Plain verbs

*brings the results together*, *the gap opens up*, *folds in the
observation*, *nudges*. Anglo-Saxon and concrete rather than Latinate.
Preserve when editing; do not "elevate" to *aggregates*, *widens*,
*incorporates*.

## C.8 Setup disclosure at hardware granularity

The sample names the exact machine, core count, and RAM. Keep this
specificity in Experiments; it costs two lines and buys reproducibility
credit.

## C.9 Defect — comma splices

Two independent clauses joined by a comma. In the sample:

> *WiSARD operates on binary inputs, each numeric feature is first
> converted into bits through a thermometer encoding.*

Fix: full stop, semicolon, or subordinate the first clause (*Because
WiSARD operates on binary inputs, each numeric feature is…*).

**Check:** search for `, each`, `, this`, `, it`, `, the` followed by a
finite verb.

## C.10 Defect — the 50-word colon sentence

The signature over-long construction: a main clause, a colon, then an
elaboration carrying two or three further subordinate clauses. The
KalWeNN definition opening Section 3 of the sample runs past sixty
words.

The colon itself is fine and characteristic (§C.1 depends on it). The
defect is the load after it.

**Check:** any sentence over ~40 words with a colon. Split at the
colon; make the elaboration its own sentence or two.

## C.11 Defect — connective monotony

*Since* opens three sentences in one section of the sample. Same risk
with *However* and *Overall*.

**Check:** no sentence-initial connective repeats within a section.

## C.12 Defect — orphaned forward references

The sample's Introduction closes with *"Overall, these gains come at a
modest cost"* — but no gain has been stated; the preceding sentence
only lists what will be analysed. The antecedent was almost certainly
lost in a cut.

**Check:** every *these / those / such* refers to something already
named in the preceding two sentences. Run this specifically after any
compression pass (§7.4), which is where these are created.

## C.13 Defect — vague magnitude when the number exists

*"often by a large factor"*, *"approximately 93%"*, *"a factor of four
to five"* — while the exact figures sit in the table. §3.8 applies.
Give the number on first mention; the qualitative phrasing can follow
it, not replace it.

## C.14 Defect — promised-but-undelivered analysis

The sample's *Stability and Computational Cost Results* heading
promises cost analysis; the prose discusses stability only, leaving
train time and memory in the table uninterpreted. §5.2 requires every
table column a heading advertises to be interpreted in text.

**Check:** each heading's nouns appear as subjects in its own prose.

## C.15 Defect — passive in the setup

*"All experiments were run"*, *"Each dataset was used in full"*.
Conventional and defensible, but §3.4 prefers active. Low priority —
only worth changing if a reviewer flags it.

## C.16 Open — personification of models and results

*"Stability is where the Kalman model makes its strongest case."* *"The
gap opens up."* *"The lowest figures of the whole table belong to the
Kalman models."* Readable and distinctive, and arguably a register leak
under §3.12. Keep, ration, or cut? Current default: **keep in Results,
avoid in Abstract and Conclusion**, pending a decision. Do not resolve
without asking.

## C.17 Open — run-in headings at full paper length

The pattern suits a 6-page workshop paper. **This is now a decision to
make, not a hypothetical:** the current target is 12 LNCS pages, where
§1.7 preambles and cross-referencing favour numbered subsections. A
workable middle is numbered subsections for structure with run-in
headings inside them for the finer divisions. Ask before drafting the
Proposal, since that section sets the pattern the rest follows.

## C.18 Open — attribution within the evidence base

Which prose in the seven-author sample is the author's own is unknown.
Any rule in §C.1–C.8 may in fact be a co-author's habit.

## C.19 Structural gaps in the sample (not style)

Recorded so they are not mistaken for preferences. Each is a
requirement from Sections 1–10 absent from the sample, most plausibly
for page-budget reasons. On a paper without that constraint, the rule
applies unchanged — **which includes the current 12-page LNCS target.**
Every row below is a requirement for the current work.

| Absent | Reference |
|---|---|
| Related Work section (folded into the Introduction) | §1.1, §1.3 |
| Contributions list of refutable claims | §4.3 |
| *"The paper is divided as follows:"* outline sentence | §1.2 |
| Explicit research questions in Experiments | §4.6 |
| Any figure; architecture diagram; pseudocode | §4.5, §5.3, §5.4 |
| Public repository link | §1.5, §1.10 |
| Confidence intervals and statistical tests (σ and CV reported instead) | §1.5 |
| Decision-rationale subsection in the Proposal | §1.4 |
| Section preamble on the Experiments section | §1.7 |
| AI-assistance disclosure | Section 10 |

The Conclusion does satisfy §1.6 in substance (summary → results →
cost, then a Future Work paragraph that sketches a research path rather
than listing wishes), though not in the literal *"The paper
represented…"* phrasing.

---

# Appendix: adding new orienting-professor feedback

When a new preference emerges from a session with the orienting
professor, add it to Section 1 directly (not to a separate appendix).
Each new rule must name:

- The behavior or convention to follow.
- The professor's rationale (or the inferred rationale).
- Any conflict with rules elsewhere in this file — flag the conflict
  explicitly.

Keep Section 1 prescriptive and short. Move detailed reasoning into the
relevant deep-dive section (2–10) and back-reference it.

New style-profile observations go to Part C, not Section 1, and must
name the draft they were observed in.
