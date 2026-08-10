
# FIELD MATE — MASTER ENGINEERING CONTEXT

This document is the persistent handoff for the FieldMate project.

FieldMate is the current StarForge implementation being built as a voice-based field diagnostic assistant. It is not intended to be a generic conversational chatbot and it is not intended to be a generic RAG demo.

The central product idea is:

    Technician speaks
        ↓
    realtime speech understanding
        ↓
    diagnostic state
        ↓
    relevant technical evidence + prior cases
        ↓
    reasoning over evidence
        ↓
    next diagnostic action
        ↓
    technician performs/checks action
        ↓
    new observation
        ↓
    state evolves
        ↓
    memory may evolve
        ↓
    repeat

The system should behave like a diagnostic partner that can remember what happened, understand what has already been tested, retrieve useful historical/domain evidence, recognize contradictions, choose the next useful diagnostic action, and communicate naturally through a low-latency voice interface.

The application owns canonical state.

The LLM reasons over state and evidence.

Qdrant stores/retrieves domain memory and knowledge.

The realtime voice stack is an interface to the brain, not the brain itself.


# PROJECT ORIGIN AND CURRENT SCOPE

The project originated as a broader idea for a voice-based diagnostic assistant for technicians working on electronic/technical equipment.

The current StarForge/FieldMate version is intentionally narrowed to PC troubleshooting.

Current domain:

- Windows PC troubleshooting
- Windows hardware troubleshooting
- Windows software troubleshooting
- Windows networking troubleshooting

Current platform scope:

- Windows PCs
- Windows laptops

Current initial OEM scope:

- Lenovo
- Dell
- HP
- ASUS

Do not silently broaden this scope.

The product should be optimized deeply for the selected domain rather than superficially supporting everything.

The scope restriction is deliberate. It makes the diagnostic state, memory schema, retrieval corpus, procedures, and reasoning policy substantially more coherent.

Examples of in-scope areas include:

Hardware:
- RAM
- SSD/HDD
- cooling
- fans
- thermals
- display
- keyboard
- touchpad
- USB
- battery
- charging
- power
- motherboard-related symptoms
- device failures
- sensors where exposed through the PC/OEM stack

Software:
- Windows boot problems
- driver issues
- Windows services
- update problems
- application failures
- crashes
- BSODs
- system corruption
- permissions
- configuration
- event/log interpretation

Networking:
- Wi-Fi
- Ethernet
- DNS
- DHCP
- adapter issues
- connectivity
- driver issues
- Windows network configuration
- common router/client interaction issues

The system should remain capable of understanding natural technician language rather than requiring rigid command syntax.


# CORE PRODUCT THESIS

A conventional RAG assistant does:

    user question
        ↓
    vector search
        ↓
    documents
        ↓
    LLM answer

That is insufficient for FieldMate.

If Qdrant merely returns:

    "Here are some documents relevant to your question."

then ChatGPT plus a conventional RAG pipeline can do essentially the same thing.

FieldMate's differentiator is domain state.

The brain should represent:

- what machine is being worked on
- what OS/environment is known
- what fault identifiers exist
- what symptoms have actually been observed
- what observations have been reported
- what measurements have been taken
- which diagnostic tests have been performed
- which tests passed/failed
- what hypotheses currently exist
- which hypothesis is currently favored
- what evidence supports each hypothesis
- what evidence contradicts each hypothesis
- what action is recommended next
- what resolution has actually been confirmed
- what happened in previous similar cases
- what memories have been reinforced or deprecated

Retrieval should therefore participate in decision-making rather than simply supplying prose context.


# NON-NEGOTIABLE ARCHITECTURAL PRINCIPLES

1. The application owns canonical diagnostic state.

2. The LLM does not own canonical state.

3. Qdrant is a real Qdrant Cloud dependency, not a homemade vector database.

4. Qdrant represents domain knowledge and evolving memory, not merely generic documents.

5. Memory must have provenance.

6. Memory confidence must evolve from evidence.

7. Contradictions must be preserved rather than silently erased.

8. Weak evidence must not automatically become a generalized pattern.

9. Confirmed resolutions are materially stronger than proposed resolutions.

10. State transitions must be deterministic and validated.

11. Events are immutable.

12. Event application is idempotent.

13. Event ID collisions are errors.

14. Stale turns/generations must not mutate newer state.

15. Failed state transitions must roll back atomically.

16. Speculative retrieval must never block the conversational path.

17. Partial speech is a signal for speculation, not an instruction to query Qdrant on every partial.

18. General/non-technical requests should bypass retrieval and route directly to the LLM.

19. Routine hardcoded filler speech should be avoided because it increases perceived latency.

20. Hardcoded Rime filler should exist only as an exceptional fallback for prolonged latency/dead air.

21. Streaming is preferred wherever possible.

22. Interruptions must cancel stale response work.

23. Old asynchronous work must never overwrite newer diagnostic state.

24. Retrieved evidence must be normalized before entering the reasoning context.

25. Contradictory evidence must remain visible to reasoning.

26. Retrieval should be state-aware.

27. Context should be budgeted rather than dumping every result into the LLM.

28. Latency must be measured by stage and by tail behavior, not only averages.

29. Existing passing tests are architectural contracts.

30. Do not replace a working component without first understanding its behavior.

31. Do not silently expand scope.

32. Do not hardcode secrets.

33. Prefer official/current vendor documentation when implementing vendor APIs.

34. Make changes in coherent vertical slices and test after each slice.


# SYSTEM OVERVIEW

The intended full system is:

Technician
    ↓
LiveKit realtime transport
    ↓
Deepgram Flux streaming STT
    ↓
Transcript / partial / EOT events
    ↓
Intent + observation routing
    ↓
Query Stabilizer
    ↓
Diagnostic State
    ↓
Retrieval Orchestrator
    ↓
Qdrant
    ↓
Evidence / Context Intelligence
    ↓
Diagnostic Reasoning
    ↓
Groq
    ↓
structured decision + streaming response
    ↓
Rime
    ↓
LiveKit audio
    ↓
Technician

Important: this is a conceptual data flow, not a requirement that every component be physically implemented in one module.

The brain should be independently testable without the voice layer.

The voice layer should be able to consume brain outputs without knowing internal Qdrant details.


# BRAIN ARCHITECTURE

The brain should be organized approximately as:

brain/
    state/
        models.py
        events.py
        engine.py
        ...
    memory/
        models.py
        evolution.py
        identity.py
        case_extraction.py
        ...
    qdrant/
        config.py
        bootstrap.py
        repository.py
        ...
    retrieval/
        planner.py
        orchestrator.py
        context.py
        stabilizer.py
        evidence.py
        ...
    diagnosis/
        hypotheses.py
        planner.py
        engine.py
        ...
    llm/
        groq.py
        schemas.py
        ...
    ...

This is a responsibility boundary, not an excuse to create needless abstractions.

Keep low-level Qdrant plumbing separate from domain memory evolution.

Keep state transitions separate from LLM generation.

Keep realtime transport separate from domain reasoning.


# DIAGNOSTIC STATE — CURRENT MODEL

The current DiagnosticState concept includes:

    equipment: EquipmentState
    fault_codes: list[str]
    symptoms: list[Observation]
    observations: list[Observation]
    measurements: list[Measurement]
    tests: list[DiagnosticTest]
    hypotheses: list[Hypothesis]
    current_hypothesis: str | None
    next_recommended_action: str | None
    confirmed_resolution: str | None
    case_status: str
    updated_at: datetime

The exact Python model is authoritative in the source tree. This document describes intended semantics.

Equipment state should eventually be able to represent relevant Windows/OEM context such as:

- OEM
- model
- family
- serial where appropriate
- Windows version/build where relevant
- subsystem/component
- relevant hardware/software identifiers

Do not put every possible environmental field into state just because it exists. Store what materially affects diagnosis.

Symptoms are user/technician observations.

Observations are factual findings.

Measurements are structured readings and should have units when applicable.

Tests represent diagnostic actions that have explicit lifecycle state.

Hypotheses represent possible causes, not confirmed facts.

The current hypothesis is a selected working hypothesis, not automatically truth.

Next recommended action is a recommendation, not proof.

Confirmed resolution means the technician/system has evidence that a corrective action actually resolved the case.

Case status should distinguish an open diagnostic case from a resolved/closed case.


# STATE ENGINE CONTRACT

The state engine is one of the most important pieces of the brain.

It must:

- accept domain events
- validate them
- reject invalid transitions
- apply them deterministically
- maintain an immutable event log
- enforce event idempotency
- detect event ID collisions
- enforce turn/generation ordering
- roll back all mutations on failure

Do not allow random application code to directly mutate canonical state when an event transition exists for that operation.

The state engine should be the gatekeeper for diagnostic state changes.

A state transition should either fully succeed or leave the prior state exactly as it was.

Atomicity covers:

- diagnostic state
- event log
- turn
- generation
- related transition bookkeeping

If a transition fails halfway through, no partial mutation may survive.


# EVENTS AND IDEMPOTENCY

Events should have stable identity.

Applying the same event twice must not duplicate its effect.

Applying two events with the same event ID but different content must fail.

This is required because realtime systems can duplicate, reorder, retry, or race event delivery.

The engine has already been tested for:

- initial event
- duplicate event idempotency
- event ID collision
- new turn/generation
- stale turn
- stale generation
- future generation
- stale event after future generation

Do not weaken these rules to make an integration test easier.

If the voice layer produces a duplicate observation event, state must remain correct.

If an old async retrieval result arrives after a newer turn, it must not mutate the current state.


# STATE ADVERSARIAL TEST CONTRACTS

The adversarial state suite has already verified:

COMPLETE NONEXISTENT TEST: PASS
UPDATE NONEXISTENT HYPOTHESIS: PASS
EMPTY FAULT CODE: PASS
MISSING FAULT CODE: PASS
INVALID MEASUREMENT: PASS
MISSING MEASUREMENT UNIT: PASS
START VALID TEST: PASS
DUPLICATE TEST START: PASS
COMPLETE VALID TEST: PASS
DUPLICATE TEST COMPLETION: PASS
CREATE HYPOTHESIS: PASS
UPDATE HYPOTHESIS: PASS
EVENT IMMUTABILITY: PASS

These tests are not disposable.

They encode safety and correctness assumptions about the domain state.


# STATE ATOMICITY TEST CONTRACTS

The atomicity suite has passed:

FAILURE CAUGHT: PASS
STATE ROLLBACK: PASS
EVENT LOG ROLLBACK: PASS
TURN ROLLBACK: PASS
GENERATION ROLLBACK: PASS
PARTIAL MUTATION REMOVED: PASS
POST-ROLLBACK RECOVERY: PASS

A future refactor must preserve all of these properties.


# MEMORY MODEL

Memory is a first-class domain object.

Memory should be able to represent:

- identity
- memory type
- equipment scope
- fault/symptom scope
- evidence
- provenance
- confidence
- verification
- resolution
- contradiction state
- deprecation
- evolution history
- timestamps
- source/case references

Memory types should remain explicit.

Potential categories include:

- case
- resolution
- procedure
- equipment-specific knowledge
- fault relationship
- diagnostic observation
- pattern
- user/technician-specific memory where appropriate

Do not flatten all memories into one undifferentiated text blob.

Different memory types have different evidentiary strength and retrieval meaning.


# MEMORY EVIDENCE AND PROVENANCE

Evidence is central to memory quality.

A memory should be able to answer:

Where did this claim come from?

Was it:

- observed directly?
- measured?
- tested?
- documented by an authoritative source?
- inferred?
- confirmed by a successful resolution?
- contradicted by another case?

Evidence should retain enough provenance to audit the memory.

The memory tests have already passed:

MEMORY CREATION
EVIDENCE PROVENANCE
RESOLUTION TRACKING
MEMORY VERIFICATION
CONTRADICTION TRACKING
MEMORY DEPRECATION
INVALID CONFIDENCE
INVALID EVIDENCE

Do not simplify memory objects by removing provenance.


# MEMORY EVOLUTION

Memory evolution is deliberate.

A useful conceptual lifecycle:

candidate
    ↓
observed
    ↓
supported
    ↓
confirmed
    ↓
reinforced
    ↓
high-confidence pattern

Contradiction may appear at any point.

Explicit deprecation should preserve history rather than erase it.

The evolution suite has passed:

INITIAL CANDIDATE: PASS
FIRST CONFIRMED CASE: PASS
MEMORY PROMOTION: PASS
CONFIDENCE EVOLUTION: PASS
MEMORY REINFORCEMENT: PASS
CONTRADICTION PRESERVED: PASS
EVIDENCE HISTORY: PASS
EXPLICIT DEPRECATION: PASS

Promotion must be conservative.

A single successful case can create a case-specific memory.

A single case should not automatically become a universal rule.

Repeated independent confirmed evidence is stronger than repeated copies of the same source.


# MEMORY IDENTITY

Logical memory identity must be deterministic.

The identity suite has passed:

SAME MEMORY IDENTITY: PASS
DIFFERENT EQUIPMENT: PASS
DIFFERENT FAULT: PASS
DIFFERENT MEMORY TYPE: PASS
DETERMINISTIC IDENTITY: PASS
FAULT ORDER INDEPENDENCE: PASS

Equivalent logical memory should produce the same identity.

Ordering of equivalent fault identifiers should not accidentally change logical identity.

Different equipment, fault, or memory type should create distinct logical identities where the domain semantics require it.

Qdrant point IDs may be deterministic UUIDs derived from this logical identity.


# CASE EXTRACTION

Completed diagnostic cases should produce structured memory candidates.

The case extraction suite has passed:

CASE EXTRACTION: PASS
MEMORY CATEGORIES: PASS
CONSERVATIVE PROMOTION: PASS
RESOLUTION MEMORY: PASS
PATTERN NOT PREMATURELY CREATED: PASS

The extractor should distinguish:

case-specific fact
versus
reusable generalized pattern.

A confirmed resolution is especially important.

Example:

Observed:
"XJ-420 reports ERR-17 and overheats."

Confirmed action:
"Replacing the temperature sensor resolved the fault."

This can produce a strong case/resolution memory.

It should not automatically produce:
"ERR-17 always means temperature sensor failure."

That would be premature generalization.


# RESOLUTION SEMANTICS

The state model distinguishes proposed from confirmed resolution.

RESOLUTION_PROPOSED means a corrective action has been proposed.

RESOLUTION_CONFIRMED means the resolution has been explicitly confirmed.

The confirmed resolution should be represented in DiagnosticState and case extraction.

A confirmed resolution should not be inferred merely because an action was recommended.

The distinction is important for memory quality and confidence evolution.


# QDRANT IS THE REAL DATABASE

Qdrant Cloud is the actual vector database.

Do not replace it with a homemade approximation.

Do not implement a Python list of vectors and call it Qdrant.

Use Qdrant's real capabilities where appropriate.

The system has already verified:

- Qdrant Cloud connectivity
- collection creation/bootstrap
- payload indexes
- memory upsert
- Cloud Inference embeddings
- dense retrieval
- sparse/BM25 retrieval
- hybrid retrieval
- Qdrant → context → Groq grounding

The memory collection currently uses:

Collection:
    fieldmate_memory

Dense model:
    sentence-transformers/all-minilm-l6-v2

Sparse model:
    qdrant/bm25

These are current project configuration values and should not be changed casually.


# QDRANT DATA MODEL

Qdrant points should contain:

- deterministic logical memory ID / point ID
- vector representations
- structured payload

Useful payload dimensions include:

- memory_id
- memory_type
- equipment OEM
- equipment family
- equipment model
- serial where appropriate
- OS
- Windows version/build where relevant
- system
- subsystem
- component
- fault_code
- symptom
- status
- confidence
- verification status
- evidence references
- provenance
- case ID
- source
- scope
- owner/user scope where appropriate
- created_at
- updated_at
- contradiction state
- deprecation state

Use payload filtering for structured constraints.

Do not force every structured field into the vector text.

Create payload indexes for high-value filter fields.

Use native Qdrant capabilities rather than rebuilding them in Python.


# QDRANT RETRIEVAL STRATEGY

Current routing:

Semantic query:
    dense

Fault identifier:
    sparse

Equipment + fault context:
    hybrid

The retrieval planner produces a RetrievalPlan including mode and reason.

Verified routing examples:

"The machine is overheating badly."
    mode = dense
    reason = semantic_query

"The machine shows ERR-17."
    mode = sparse
    reason = fault_identifier

"The machine is overheating and showing ERR-17."
with equipment model XJ-420 and fault ERR-17
    mode = hybrid
    reason = equipment_and_fault_context

The routing reason is useful for observability and future adaptive behavior.


# QDRANT HYBRID SEARCH

Hybrid retrieval is valuable because diagnostic queries often contain both:

- semantic symptom/context
- exact identifiers

For example:

"XJ-420 is overheating and showing ERR-17."

Dense search can capture semantic relationships.

Sparse/BM25 search can strongly capture exact identifiers such as:

- ERR-17
- WHEA-Logger
- specific Windows error names
- OEM model numbers
- driver identifiers

Hybrid retrieval should combine these signals using Qdrant's native query capabilities/fusion where appropriate.

Do not assume hybrid is always slower.

Observed latency varies substantially between runs.

Therefore:

- instrument it
- bound it
- use it when information value justifies it
- do not hardcode simplistic "hybrid is bad" rules.


# QDRANT CLOUD INFERENCE

Cloud Inference has already been tested successfully.

Verified flow:

TEXT
→ embedding
→ Qdrant search
→ results

Warm query measurements have shown first-query warm-up effects and subsequent lower latency.

This means latency tuning should account for:

- cold start/warm-up
- network
- embedding generation
- Qdrant search
- result conversion

Do not assume a single benchmark is representative.


# QDRANT MEMORY TEST

A Qdrant memory integration test successfully demonstrated:

INITIALIZING QDRANT
QDRANT READY
UPSERTING MEMORY
POINT ID generated deterministically
UPSERT successful
HYBRID SEARCH successful
PAYLOAD VERIFIED: PASS
HYBRID RETRIEVAL: PASS

A representative memory:

"On an XJ-420 machine showing ERR-17 with overheating, replacing the temperature sensor resolved the fault."

This is a case/resolution style memory and demonstrates the intended Qdrant role.


# QDRANT LATENCY OBSERVATIONS

Observed latency has varied.

An earlier run showed:

DENSE ONLY
~809.94 ms

BM25 ONLY
~270.86 ms

HYBRID
~3108.74 ms

Warm hybrid runs varied approximately:

~3899 ms
~2268 ms
~3914 ms
~1364 ms
~6749 ms

Later runs showed significantly better results.

Therefore latency is variable and likely influenced by cold/warm inference, network, service load, query path, and implementation details.

Do not treat old benchmark values as fixed performance.

Instrument every stage before optimizing blindly.


# REPOSITORY LAYER

Qdrant repository responsibilities:

- Qdrant client lifecycle
- configuration
- collection ensure/bootstrap
- payload indexes
- upsert
- dense search
- sparse search
- hybrid search
- filter construction
- point conversion
- error handling
- observability
- shutdown

The repository should not decide diagnostic business rules.

Memory evolution belongs in the memory/domain layer.

Retrieval policy belongs in planner/orchestrator.

Reasoning belongs in diagnosis/LLM layers.


# RETRIEVAL ORCHESTRATOR

The RetrievalOrchestrator is responsible for:

- planning retrieval mode
- constructing state-aware filters
- checking completed speculative retrieval
- performing bounded normal retrieval
- launching speculative prefetch
- maintaining prefetch cache entries
- preventing duplicate active prefetches
- respecting TTL
- cancelling stale/background work
- returning normalized RetrievalResult

RetrievalResult concept:

- context
- plan
- latency_ms
- timed_out
- prefetched

The orchestrator is a latency/control layer, not a domain reasoning engine.


# HOT PATH VS PREFETCH

There are two distinct latency budgets.

HOT PATH:
The user is waiting for the answer.

This must be bounded.

PREFETCH:
Work happening speculatively while the technician is still speaking.

This may have a longer budget.

Critical invariant:

    The hot path NEVER awaits an unfinished prefetch.

If a prefetch is complete:
    consume it.

If still running:
    do not wait.

If stale:
    discard.

If failed:
    discard.

If cancelled:
    discard.

Then the hot path can perform normal bounded retrieval.

This distinction was necessary because the earlier implementation used the same 900 ms budget for normal retrieval and speculative retrieval. That caused a test race when retrieval took longer than 900 ms.

After increasing the diagnostic test budget, successful measured retrieval included approximately:

Dense: 439.37 ms
Sparse: 331.70 ms
Hybrid: 686.91 ms

And completed prefetch consumption:
0.21 ms

That 0.21 ms is an important success condition: retrieval happened before final consumption, so the conversational path can reuse it essentially immediately.


# SPECULATIVE PREFETCH DESIGN

The intended voice-time behavior:

Technician says:
    "My Dell laptop keeps disconnecting from Wi-Fi..."

Flux emits partial transcript.

The system may launch speculative retrieval.

The technician continues:
    "...after waking from sleep."

The query stabilizer decides whether the new information materially changes the search.

If not:
    keep existing speculative result.

If yes:
    cancel/replace stale speculation.

At end-of-turn:
    if the correct prefetch is complete:
        consume it.

Otherwise:
    perform bounded retrieval.

The prefetch is an optimization, never a correctness dependency.


# QUERY STABILIZER — REQUIRED NEXT LAYER

Partial transcripts should not each become retrieval queries.

Bad behavior:

"My"
"My Dell"
"My Dell laptop"
"My Dell laptop keeps"
"My Dell laptop keeps disconnecting"
"My Dell laptop keeps disconnecting from"
...

Each one causing Qdrant retrieval would waste compute and create latency/load.

Desired behavior:

partial transcript
    ↓
normalize
    ↓
compare with prior stable candidate
    ↓
detect meaningful delta
    ↓
if meaningful:
    speculate
    else:
    ignore

Important triggers:

- newly detected OEM
- newly detected model
- newly detected fault code
- newly detected Windows error
- newly detected component
- newly detected symptom
- major change in diagnostic intent

Do not search merely because the transcript changed by one word.

Fault identifiers deserve special priority because sparse retrieval can exploit exact lexical matches.


# NON-TECHNICAL FAST PATH

Not every utterance belongs in the diagnostic retrieval pipeline.

Examples:

"Hey, what can you do?"
"Thanks."
"Okay."
"Explain that again."
"Can you repeat the last step?"

These can bypass Qdrant.

Desired routing:

input
    ↓
cheap intent classification
    ↓
technical diagnostic?
    ├── yes → state/retrieval/reasoning
    └── no  → direct LLM response

The classifier should be cheap.

Do not invoke a heavyweight reasoning model just to determine that "thanks" is non-technical.

This is a perceived-latency optimization as well as a cost optimization.


# CONTEXT INTELLIGENCE

Context Intelligence is the next major brain layer.

It should combine:

- current DiagnosticState
- current user observation
- retrieved memories
- technical knowledge
- previous cases
- evidence provenance
- confidence
- verification
- contradictions
- current hypotheses
- completed tests
- unresolved questions

Pipeline:

retrieved candidates
    ↓
deduplicate
    ↓
filter against state
    ↓
score relevance/evidence
    ↓
classify support vs contradiction
    ↓
apply confidence/evidence weighting
    ↓
budget context
    ↓
produce structured reasoning context

The output should be compact and explicit rather than a giant concatenated text blob.


# EVIDENCE NORMALIZATION

Qdrant result objects should be converted into a domain-level Evidence representation.

Conceptual fields:

- evidence_id
- memory_id
- memory_type
- text/content
- source
- source_type
- equipment scope
- fault scope
- relevance score
- confidence
- verification
- provenance
- supporting/contradicting relation
- case reference
- timestamp
- retrieval mode
- retrieval score

This prevents Groq from having to understand raw Qdrant implementation details.

The LLM should receive evidence semantics, not database plumbing.


# CONTEXT BUDGETING

Do not send every retrieved memory to Groq.

Context should be selected based on:

- direct relevance
- equipment match
- fault match
- symptom match
- verification strength
- confirmed resolution strength
- recency where relevant
- contradiction importance
- source authority
- diversity of evidence

A contradictory high-quality memory should be retained even if it has a slightly lower semantic score.

A low-quality duplicate of an already selected memory should be removed.

The goal is maximum diagnostic information per token.


# CONTRADICTION HANDLING

Contradictions are valuable.

Example:

Memory A:
    WHEA-related symptom is associated with a hardware issue.

Memory B:
    Similar symptom in a different case was caused by a driver problem.

The context layer should preserve:

Supporting evidence:
    ...

Alternative/contradictory evidence:
    ...

Evidence strength:
    ...

Confirmed case count:
    ...

The reasoning engine can then choose a discriminating test.

Never silently collapse contradictions into one answer merely because a vector search produced one higher score.


# DIAGNOSTIC REASONING ENGINE

The reasoning engine should answer:

What do we know?

What do we not know?

What has already been tested?

What hypotheses remain?

Which hypothesis currently has the strongest evidence?

What observation/test would best discriminate between competing hypotheses?

What is the safest useful next action?

What should be communicated to the technician now?

The reasoning engine should prefer information-gain-oriented diagnostic steps.

Example:

Hypothesis A:
    Wi-Fi adapter driver issue

Hypothesis B:
    access point/DHCP issue

Already tested:
    another Wi-Fi network works

That observation changes the hypothesis balance.

The system should not blindly repeat already-completed tests.


# GROQ ROLE

Groq is the LLM reasoning/generation layer.

Groq should receive:

- current state summary
- relevant evidence
- contradictions
- hypotheses
- completed tests
- current observation
- task/instruction

The LLM output should be structured enough for the application to validate.

Conceptual decision object:

- response
- selected hypothesis
- confidence
- state updates
- recommended next action
- clarification needed
- evidence references
- resolution proposal/confirmation where applicable

The application validates state updates before applying them.

Never allow raw LLM text to directly mutate canonical state.


# GROQ STREAMING

The project has already tested Groq streaming successfully.

A warm test using llama-3.1-8b-instant produced observed TTFTs such as:

~222 ms
~145 ms
~258 ms
~138 ms
~406 ms

Total generation varied roughly from a few hundred milliseconds to around one second.

The system should stream.

Do not wait for the entire generated answer before starting downstream speech when the voice architecture can safely consume partial output.

The exact production model should be selected based on actual structured-output quality, latency, context needs, and current availability rather than assumptions.


# GROUNDED QDRANT → GROQ FLOW

The Qdrant → Groq integration has already been verified.

Test question:
"My XJ-420 machine is overheating and showing fault E17. What should I check?"

Retrieved context included:

"The XJ-420 machine reports fault E17 when the motor temperature becomes too high."

and:

"If the XJ-420 overheats, inspect the coolant level, cooling system, radiator and temperature sensor."

Groq then produced a grounded response focused on checking:

- coolant level
- cooling system
- radiator
- temperature sensor

The test passed:

QDRANT → CONTEXT → GROQ
GROUNDED STREAMING RESPONSE WORKS

This proves the core grounding path exists.


# VOICE STACK

Selected realtime stack:

- LiveKit
- Deepgram Flux
- Groq
- Rime

Conceptual responsibilities:

LiveKit:
    realtime transport/session/audio/data

Deepgram Flux:
    streaming speech recognition and turn/EOT-related signals

Brain:
    state, retrieval, memory, diagnosis

Groq:
    reasoning and response generation

Rime:
    streaming TTS

The brain must remain testable independently of the voice transport.


# LIVEKIT ROLE

LiveKit should handle realtime communication/session concerns.

The brain should not become coupled to low-level WebRTC/media details.

LiveKit events should be translated into brain-level events such as:

- partial transcript
- finalized transcript
- turn started
- turn ended
- interruption
- cancellation
- session start/end

The brain should consume those semantic events.

This separation allows unit testing of diagnostic logic without a microphone or network connection.


# DEEPGRAM FLUX ROLE

Deepgram Flux is the streaming STT/turn-detection layer.

The system should exploit partial transcripts for speculation but must not overreact to every partial.

Useful signals include:

- partial transcript
- stable transcript
- end-of-turn
- interruption/turn transition

The query stabilizer sits between raw streaming transcript events and retrieval.

The final diagnostic observation should be based on finalized/stable turn information, while speculative retrieval may use earlier information.


# RIME ROLE

Rime is the TTS layer.

The desired behavior is streaming audio.

The system should be able to:

- begin TTS as useful response text becomes available
- cancel/stop when technician interrupts
- flush/finalize appropriately
- avoid speaking stale responses
- avoid routine filler

Hardcoded filler should be rare.

A short fallback can be used if a genuinely long delay creates dead air, but it should not be the normal path because filler itself adds speech time and perceived latency.


# INTERRUPTION MODEL

Realtime interruption is a first-class condition.

When technician speech interrupts FieldMate:

1. stop/cancel Rime output
2. stop/cancel unnecessary Groq generation
3. invalidate stale response generation
4. preserve valid diagnostic state
5. process new technician observation
6. launch updated speculation if useful
7. never let old asynchronous work write newer state

Turn/generation IDs exist specifically to help enforce this.


# PERCEIVED LATENCY

The optimization target is not simply API latency.

The target is perceived time from:

technician finishes meaningful utterance
→ technician hears useful FieldMate response

Useful optimizations:

- partial STT
- speculative Qdrant retrieval
- cheap non-technical bypass
- sparse routing for identifiers
- context compression
- streaming Groq
- streaming Rime
- interruption cancellation
- no redundant retrieval
- no unnecessary filler
- reuse completed speculative results

Example desired path:

Technician is still speaking
    ↓
Qdrant retrieval already running
    ↓
technician finishes
    ↓
prefetch result available
    ↓
Groq starts immediately
    ↓
first response tokens
    ↓
Rime begins speaking

This is much better than:

technician finishes
    ↓
retrieve
    ↓
wait
    ↓
LLM
    ↓
wait
    ↓
TTS
    ↓
speak


# LATENCY INSTRUMENTATION

Every major stage should eventually be timed separately.

Desired trace:

STT partial latency
EOT latency
query stabilization
planner
filter construction
embedding/query construction
Qdrant request
Qdrant server/search
result conversion
context construction
Groq TTFT
Groq generation
Rime first audio
TTS total
end-to-end response

Example:

PLAN:              0.05 ms
FILTER:            0.02 ms
QDRANT:          312.40 ms
CONTEXT:           0.30 ms
GROQ TTFT:       145.00 ms
RIME FIRST AUDIO: ...
TOTAL:             ...

Track distributions, especially p95/p99, rather than only averages.


# FAILURE AND FALLBACK PHILOSOPHY

Network services can fail.

The system should degrade gracefully.

Potential failure classes:

Qdrant unavailable:
    use bounded failure behavior and tell reasoning layer evidence is unavailable.

Groq unavailable:
    provide appropriate fallback if possible; do not fabricate diagnosis.

Rime unavailable:
    preserve text response path if the product environment permits.

Deepgram unavailable:
    surface session failure rather than inventing transcript.

LiveKit failure:
    terminate/repair session appropriately.

Prefetch failure:
    ignore and perform normal retrieval.

Prefetch timeout:
    ignore and perform normal retrieval.

Stale async result:
    discard.

The system must never replace missing evidence with invented evidence.


# SAFETY AND DIAGNOSTIC HONESTY

The system must distinguish:

OBSERVED
    technician/system directly observed it

MEASURED
    numeric or structured measurement exists

TESTED
    diagnostic test was performed

INFERRED
    reasoning suggests it

PROPOSED
    action/hypothesis is suggested

CONFIRMED
    evidence confirms it

RESOLVED
    resolution was actually verified

The assistant must not turn inferred/proposed claims into confirmed facts.

If evidence is insufficient, ask for a useful next check.

For hardware/electrical situations, avoid unsafe instructions and stay within appropriate troubleshooting boundaries.


# TEST SUITE — CURRENT CONTRACTS

Existing passing tests should be preserved.

State tests:
- state test
- adversarial test
- idempotency test
- atomicity test

Memory tests:
- memory test
- evolution test
- case extraction test
- identity test

Qdrant tests:
- bootstrap
- memory integration
- latency benchmark

Retrieval:
- planner/routing test
- orchestrator test

Do not remove a test merely because a new implementation makes it inconvenient.

If a test is wrong, document why and replace it with an equivalent or stronger invariant.


# TEST RESULTS — STATE

Verified:

STATE:
PASS

ADVERSARIAL:
PASS

IDEMPOTENCY:
PASS

ATOMICITY:
PASS

The adversarial suite specifically passed:
- invalid/nonexistent test handling
- invalid hypothesis handling
- invalid fault codes
- invalid measurements
- test lifecycle
- hypothesis lifecycle
- event immutability

The idempotency suite passed:
- duplicate event
- collision
- new turn/generation
- stale turn
- stale generation
- future generation
- stale-after-future-generation

The atomicity suite passed:
- failure caught
- state rollback
- event log rollback
- turn rollback
- generation rollback
- partial mutation removal
- recovery.


# TEST RESULTS — MEMORY

Verified:

MEMORY:
PASS

MEMORY EVOLUTION:
PASS

CASE EXTRACTION:
PASS

IDENTITY:
PASS

Important demonstrated properties:
- evidence provenance
- resolution tracking
- verification
- contradiction tracking
- deprecation
- confidence validation
- candidate → confirmed → promoted evolution
- reinforcement
- contradiction preservation
- evidence history
- conservative case extraction
- no premature generalized pattern
- deterministic identity
- fault-order independence


# TEST RESULTS — QDRANT

Verified:

QDRANT BOOTSTRAP:
PASS

QDRANT MEMORY:
PASS

HYBRID RETRIEVAL:
PASS

QDRANT → GROQ:
PASS

Cloud Inference:
PASS

The Qdrant memory integration verified:
- collection ready
- memory upsert
- deterministic point ID
- payload
- hybrid retrieval

The Qdrant → Groq test verified grounded streaming response.


# TEST RESULTS — RETRIEVAL ORCHESTRATOR

Latest successful orchestrator run:

SEMANTIC QUERY
MODE: dense
REASON: semantic_query
LATENCY: 439.37 ms
SEMANTIC ROUTING: PASS

FAULT QUERY
MODE: sparse
REASON: fault_identifier
LATENCY: 331.70 ms
SPARSE ROUTING: PASS

HYBRID QUERY
MODE: hybrid
REASON: equipment_and_fault_context
LATENCY: 686.91 ms
HYBRID ROUTING: PASS

SPECULATIVE PREFETCH
CONSUME LATENCY: 0.21 ms
PREFETCHED: True
SPECULATIVE RETRIEVAL: PASS

This is an important milestone.

It proves the orchestrator can perform adaptive routing and can consume a completed speculative result without blocking for the retrieval itself.


# PROJECT DIRECTORY HISTORY / PACKAGING LESSON

An early project layout accidentally contained:

src/fieldmate/src/fieldmate/brain/

This caused:

ModuleNotFoundError: No module named 'fieldmate.brain'

The brain directory was moved to:

src/fieldmate/brain/

and the nested src/fieldmate directory was removed.

Current package layout should remain conventional and compatible with the project's existing pyproject configuration.

Do not recreate nested package roots accidentally.


# DEVELOPMENT ENVIRONMENT

The project uses Python and uv.

Observed environment:
- Python 3.14.x
- uv
- package installed as fieldmate
- Qdrant client
- Groq SDK
- dotenv configuration

Commands generally used:

uv run python -m <module>

uv run python -m py_compile <file>

Run commands from:

/home/hdd/projects/fieldmate

Do not assume a shell command was successful simply because it was issued. Inspect output.


# CONFIGURATION AND SECRETS

API credentials must remain outside source control.

Likely environment values include Qdrant credentials, Groq credentials, and voice provider credentials.

Use .env locally where appropriate.

Ensure .env is gitignored.

Never put API keys into:
- AGENTS.md
- source files
- tests
- documentation
- commit history
- prompts

The coding agent may inspect environment variable names but should not print secret values.


# DOCUMENTATION STRATEGY

Project-specific architecture should live in the repository.

Recommended structure:

AGENTS.md
docs/
    architecture/
        brain.md
        state.md
        memory.md
        retrieval.md
        voice.md
        latency.md
        security.md
    vendor/
        qdrant/
        groq/
        livekit/
        deepgram/
        rime/

AGENTS.md should provide the persistent high-level rules and point to deeper documents.

Vendor documentation should not be dumped wholesale into the repository.

Only retain useful/current reference material for the APIs/features actually used.

For current vendor API behavior, verify against official documentation.

The coding agent should not invent an API because an old local document says it exists.


# VENDOR DOCUMENTATION — QDRANT

Relevant Qdrant documentation topics to provide to the coding agent:

- Python client
- async Qdrant client
- collections
- vector configurations
- named vectors where applicable
- dense vectors
- sparse vectors
- BM25
- Query API
- prefetch
- fusion
- Reciprocal Rank Fusion
- filtering
- payload indexes
- search/query limits
- score thresholds
- batch operations
- upsert
- Cloud Inference
- multivector features if later relevant
- quantization if later relevant
- indexing/storage optimization
- Cloud-specific latency/performance behavior

Do not assume every advanced feature should be enabled.

Use a feature when it improves FieldMate's actual workload.

Qdrant should remain the real database and should be used natively rather than emulated.


# VENDOR DOCUMENTATION — GROQ

Relevant Groq documentation topics:

- current Python SDK
- streaming chat completions/responses
- structured output
- JSON/schema behavior
- model capabilities
- token/context limits
- latency behavior
- errors
- retries
- timeouts
- connection reuse
- production recommendations

The project has already validated Groq streaming and observed warm TTFT in the low hundreds of milliseconds for one tested model.

Do not hardcode a model forever. Benchmark and verify the current selected model before production deployment.


# VENDOR DOCUMENTATION — LIVEKIT

Relevant LiveKit topics:

- Agents
- realtime rooms
- audio tracks
- data channels/events
- session lifecycle
- turn-taking
- interruption
- cancellation
- participant state
- realtime agent pipelines
- deployment/runtime configuration

The brain should not depend on LiveKit-specific object structures more than necessary.

Translate transport events into domain/session events.


# VENDOR DOCUMENTATION — DEEPGRAM FLUX

Relevant Deepgram topics:

- Flux streaming STT
- WebSocket/session behavior
- partial transcripts
- finalized transcript
- end-of-turn behavior
- turn detection
- interruption
- connection lifecycle
- errors
- reconnect/retry behavior
- latency characteristics

Partial transcript behavior matters directly to Query Stabilizer and speculative retrieval.

Do not treat every transcript event as a completed diagnostic turn.


# VENDOR DOCUMENTATION — RIME

Relevant Rime topics:

- Coda WebSocket API
- streaming synthesis
- audio events
- flush
- EOS
- cancellation/interrupt behavior
- connection lifecycle
- latency
- voice/model configuration

Rime is downstream of reasoning.

The TTS layer must be cancellable and interruption-aware.

Do not allow stale responses to continue speaking after a new technician turn begins.


# MEMORY + QDRANT RELATIONSHIP

Qdrant is the long-lived retrieval substrate.

Memory domain objects should not be identical to raw Qdrant payloads.

Preferred separation:

Memory domain object
    ↓
memory serializer/index representation
    ↓
Qdrant point

Qdrant result
    ↓
memory/evidence reconstruction
    ↓
Context Intelligence

This makes it possible to evolve the domain model without coupling every business rule to Qdrant internals.


# LIVE STATE + MEMORY RELATIONSHIP

Current DiagnosticState and long-lived Memory are different concepts.

DiagnosticState:
    what is happening in the current case right now

Memory:
    what the system learned/retained from prior cases or authoritative knowledge

Current state should influence retrieval.

Retrieved memory should influence reasoning.

Reasoning should produce validated state updates.

Completed cases may produce new memory candidates.

This creates the intended learning loop:

case
→ resolution
→ evidence
→ memory candidate
→ confirmation
→ promotion/reinforcement
→ future retrieval
→ better diagnosis


# USER/TECHNICIAN-SPECIFIC MEMORY

Technician/user-specific history is a planned memory category.

Examples could include:

- recurring equipment handled by a technician
- technician-specific successful troubleshooting patterns
- previous cases
- preferences in how results are communicated

However, user-specific data must be carefully scoped.

Do not leak one technician's private history into another user's diagnostic context.

Use explicit ownership/scope metadata and filters where appropriate.

Do not store sensitive personal information merely because it is available.


# EQUIPMENT MEMORY

Equipment-specific memory should be especially useful in the Windows/OEM domain.

Potential dimensions:

OEM:
    Lenovo / Dell / HP / ASUS

family/model:
    exact model or model family

OS:
    Windows version/build

component:
    e.g. Wi-Fi adapter, SSD, display

fault:
    exact code or identifier

symptom:
    natural-language symptom

resolution:
    confirmed corrective action

Evidence should distinguish an exact model-specific confirmed case from a broad OEM-level pattern.


# DIAGNOSTIC PROCEDURES AS KNOWLEDGE

Procedures are not merely text.

A procedure can contain:

- prerequisites
- target subsystem
- diagnostic purpose
- steps
- expected observation
- abnormal observation
- safety constraints
- stopping conditions
- follow-up actions
- evidence/source
- supported Windows versions/OEMs

Retrieval should be able to return procedures relevant to the current state.

The reasoning engine should select an appropriate procedure step rather than blindly quote a whole manual.


# NEXT ACTION SELECTION

A diagnostic assistant should prefer actions that reduce uncertainty.

A useful conceptual scoring model can consider:

- expected information gain
- safety
- cost/time
- reversibility
- equipment risk
- relevance to current hypotheses
- whether the action has already been performed
- evidence strength

For example, if two hypotheses explain overheating, a temperature measurement may be more useful than immediately recommending component replacement.

Do not optimize only for shortest answer.

Optimize for useful diagnosis.


# HYPOTHESIS MANAGEMENT

Hypotheses should have:

- identity
- description
- supporting evidence
- contradicting evidence
- confidence
- status
- related tests
- timestamps

Potential status concepts:

- candidate
- active
- weakened
- supported
- rejected
- confirmed

The LLM may suggest hypothesis updates, but the application validates them.

A hypothesis is not a fact simply because Groq generated it confidently.


# TEST MANAGEMENT

Diagnostic tests should have explicit state.

Conceptually:

not_started
    ↓
running
    ↓
completed
    ↓
result recorded

A test should not be completed twice accidentally.

A nonexistent test cannot be completed.

Duplicate start should be rejected or handled idempotently according to event semantics.

A test result should be attached to the correct test identity and turn/generation context.

The adversarial suite already protects these semantics.


# MEASUREMENTS

Measurements should be structured.

A measurement should have:

- value
- unit where applicable
- target/metric
- timestamp
- source
- context
- optional expected range

Missing units should be rejected where the measurement requires a unit.

Invalid measurement values should be rejected.

Do not store "72" without knowing whether it means °C, %, volts, MB/s, etc. when the domain requires a unit.


# FAULT CODES

Fault/error identifiers should be normalized carefully.

Examples:

- ERR-17
- WHEA-related identifiers
- Windows event identifiers
- OEM diagnostics codes
- driver/device identifiers

Exact identifiers should influence sparse retrieval.

Normalization should handle harmless formatting differences while preserving the original identifier for provenance.

Empty or missing required fault codes should be rejected by state transitions where the event requires one.


# GENERAL CONVERSATION VS DIAGNOSTIC SESSION

FieldMate can support lightweight conversational interaction, but the diagnostic session should remain explicit.

A general conversation does not necessarily mutate DiagnosticState.

A diagnostic observation should.

This distinction prevents:

"Thanks, that worked."

from accidentally being interpreted as a new technical observation.

However, when the user says something that implies resolution, the system may classify it as a candidate resolution confirmation and require appropriate evidence/validation rather than automatically closing the case.


# SESSION LIFECYCLE

A session should conceptually include:

- session identity
- technician/user scope
- start time
- active diagnostic case
- current turn
- current generation
- state
- event log
- active retrieval work
- active LLM generation
- active TTS work
- shutdown/cancellation state

On session close:

- cancel outstanding speculative work
- cancel generation
- stop TTS
- flush/persist relevant case state/memory according to policy
- close clients cleanly


# CONCURRENCY MODEL

Realtime voice creates concurrency.

Potential simultaneous operations:

- STT partials
- EOT detection
- speculative Qdrant
- normal retrieval
- Groq streaming
- Rime streaming
- interruption
- state event application
- memory extraction

Every asynchronous operation needs a clear ownership/lifetime model.

Turn/generation IDs are essential.

A newer generation invalidates older response work.

Do not rely solely on task cancellation. Also validate generation identity before applying results.


# STALE WORK

Example:

Turn 4:
    technician asks about Wi-Fi.

Qdrant prefetch begins.

Technician interrupts and starts Turn 5:
    "Actually, the screen is black."

Turn 4 Qdrant result eventually completes.

It must NOT be applied as if it were Turn 5 evidence.

The system should compare turn/generation identity and discard stale results.

The same principle applies to:

- Groq responses
- TTS
- state updates
- memory extraction
- speculative retrieval


# DIRECT LLM PATH

The direct LLM path is for non-technical/general conversation.

It should still preserve session safety.

It should not:

- mutate diagnostic state without a state event
- create memory without explicit reason
- trigger expensive Qdrant retrieval
- start unnecessary TTS filler

It can respond naturally and quickly.


# BRAIN PIPELINE — DETAILED

A desired turn pipeline:

1. Technician audio arrives.

2. Flux produces partial transcript.

3. Query Stabilizer normalizes partial.

4. Cheap intent/technical classifier determines whether diagnostic processing is likely.

5. If technical:
       extract obvious entities
       update speculative query
       launch/reuse prefetch

6. Flux/EOT marks turn stable.

7. Final observation is converted to domain events.

8. State engine validates/applies events.

9. Retrieval orchestrator checks for matching completed prefetch.

10. If available:
        consume it.

11. Otherwise:
        perform bounded retrieval.

12. Evidence normalization converts results into domain evidence.

13. Context Intelligence:
        deduplicates
        weighs
        preserves contradictions
        budgets context.

14. Diagnostic reasoning calls Groq.

15. Structured result is validated.

16. Valid state updates become events.

17. State engine applies updates atomically.

18. Response text streams toward Rime.

19. Rime streams audio.

20. Technician may interrupt at any point.

21. On interruption:
        invalidate current response generation
        cancel TTS
        preserve valid state
        start next turn.


# MEMORY WRITE PIPELINE

A completed case should not immediately mutate long-term memory from arbitrary text.

Preferred:

case closes/resolution confirmed
    ↓
case extractor
    ↓
candidate memory objects
    ↓
validate identity/provenance
    ↓
compare with existing memory
    ↓
reinforce / create / contradict / deprecate
    ↓
Qdrant upsert/update
    ↓
evidence history retained

This is a controlled memory evolution pipeline.

Memory writes should be much more conservative than retrieval reads.


# MEMORY REINFORCEMENT

When a new confirmed case supports an existing memory:

- increment appropriate evidence/reinforcement state
- preserve the new evidence reference
- update confidence according to defined rules
- do not create a duplicate logical memory
- preserve case-specific details

If the new case contradicts the existing memory:

- preserve both
- record contradiction
- update confidence/strength appropriately
- do not silently overwrite the older memory


# DEPRECATION

Deprecation is not deletion.

A deprecated memory should remain auditable.

Possible reasons:

- source superseded
- procedure no longer applies
- repeated contradiction
- equipment generation changed
- OS version no longer supported
- memory was found erroneous

Deprecated memories should normally be excluded from ordinary retrieval unless historical context is explicitly requested, while remaining available for audit/evolution.


# RETRIEVAL FILTERING

State-aware filtering should be applied before or during Qdrant retrieval when useful.

Examples:

equipment_model = exact model
OEM = Dell
OS = Windows 11
fault_code = ERR-17
memory_type = resolution
status = confirmed
verified_only = true

Filters should narrow the candidate set without destroying useful semantic recall.

Use exact metadata filters for exact facts.

Use vector search for semantic similarity.

Use hybrid when both are valuable.


# RETRIEVAL RESULT RANKING

A raw vector score is not the final diagnostic relevance score.

The context layer may combine:

- Qdrant retrieval score
- exact fault match
- equipment match
- OS match
- component match
- memory type
- verification
- confidence
- recency
- provenance quality
- contradiction status
- case success

Do not overfit to one numeric score.

The ranking should be explainable enough to debug.


# OBSERVABILITY EVENTS

Useful internal metrics/events:

retrieval_planned
retrieval_started
retrieval_completed
retrieval_timeout
prefetch_started
prefetch_completed
prefetch_consumed
prefetch_stale
prefetch_cancelled
prefetch_failed
context_built
groq_started
groq_first_token
groq_completed
groq_cancelled
rime_started
rime_first_audio
rime_cancelled
turn_started
turn_ended
interruption
state_event_applied
state_event_rejected
memory_created
memory_reinforced
memory_contradicted
memory_deprecated

Keep sensitive transcript/content out of generic logs where not needed.


# PRODUCTION HARDENING

Before production:

- timeouts on every external call
- cancellation support
- connection reuse
- bounded concurrency
- retry only when safe
- idempotent writes
- structured errors
- metrics
- tracing
- health checks
- graceful shutdown
- secret management
- input validation
- output validation
- stale generation protection
- memory write safeguards
- Qdrant filter/index verification
- long-session testing

Do not make retries that duplicate non-idempotent operations.


# SECURITY / PRIVACY

The system may eventually contain:

- technician identity
- equipment identifiers
- serial numbers
- diagnostic history
- organization-specific knowledge

Treat these as potentially sensitive.

Scope memories correctly.

Avoid putting secrets in vector payloads.

Do not expose one user's case history to another.

Keep API keys outside the repository.

Minimize stored data.

Make deletion/deprecation policies explicit.


# CODING AGENT OPERATING RULES

When an autonomous coding agent is attached to this repository:

1. Read this document.

2. Inspect the actual repository before making changes.

3. Inspect pyproject.toml and dependency versions.

4. Run relevant existing tests before modifying behavior.

5. Understand the current implementation rather than recreating it.

6. Preserve passing tests.

7. Prefer small, coherent changes.

8. Run formatting/type/compile/test checks after changes.

9. Inspect git diff.

10. Do not change unrelated code.

11. Do not invent library APIs.

12. Verify external API behavior against current official docs.

13. Do not add a dependency unless justified.

14. Do not store secrets.

15. Do not delete project files unless explicitly required.

16. Do not replace Qdrant with an in-memory imitation.

17. Do not turn the project into generic RAG.

18. Do not make the LLM canonical state owner.

19. Do not block on speculative retrieval.

20. Do not query Qdrant on every STT partial.

21. Do not erase contradictions.

22. Do not prematurely promote memories.

23. Do not weaken state invariants to make tests pass.

24. If a test fails, identify whether the implementation or test assumption is wrong before changing either.

25. If architecture must change, document why.


# AUTONOMOUS AGENT WORKFLOW

Preferred loop:

inspect
→ understand
→ test baseline
→ modify
→ compile
→ run focused tests
→ run broader tests
→ inspect diff
→ benchmark
→ document meaningful change

For larger changes:

1. Write/adjust a focused invariant test.

2. Implement.

3. Run focused test.

4. Run related suite.

5. Run full suite.

6. Benchmark if latency-sensitive.

7. Inspect logs/trace.

8. Only then proceed to next layer.


# WHAT IS ALREADY BUILT

The project has progressed through these layers:

1. Groq connectivity and streaming test.

2. Qdrant Cloud connectivity.

3. Qdrant Cloud Inference embedding/search test.

4. Qdrant → Groq grounded response.

5. Diagnostic state engine.

6. State adversarial validation.

7. State idempotency.

8. State atomicity.

9. Memory model.

10. Memory evolution.

11. Case extraction.

12. Deterministic memory identity.

13. Qdrant memory collection/bootstrap.

14. Qdrant memory repository.

15. Dense/sparse/hybrid retrieval.

16. Retrieval planner.

17. Retrieval orchestrator.

18. Speculative prefetch.

19. Completed-prefetch consumption.

The next work is to integrate these into a coherent intelligence pipeline rather than continuing isolated feature tests.


# WHAT REMAINS

Major remaining work:

Brain:
- query stabilizer
- evidence normalization
- context intelligence
- contradiction-aware context
- context budget
- state-aware retrieval
- diagnostic reasoning engine
- structured Groq decision schema
- validated LLM state updates
- adaptive latency budgets
- observability

Memory:
- production memory write pipeline
- case-to-memory integration
- reinforcement in Qdrant
- contradiction update pipeline
- deprecation filtering
- prior-case retrieval

Voice:
- LiveKit integration
- Flux integration
- partial transcript pipeline
- EOT integration
- speculative retrieval during speech
- Groq streaming integration
- Rime streaming
- interruption/cancellation

Hardening:
- concurrency
- failure recovery
- network errors
- stale work
- long sessions
- performance benchmarks
- production logging/tracing
- security/privacy


# DETAILED ROADMAP

PHASE 1 — BRAIN COMPLETION

1. Query Stabilizer
2. Evidence domain model
3. Evidence normalization
4. Context Intelligence
5. contradiction handling
6. context budget
7. state-aware retrieval
8. retrieval instrumentation
9. adaptive retrieval budgets

PHASE 2 — DIAGNOSTIC REASONING

1. hypothesis manager
2. diagnostic planner
3. next-test selection
4. structured Groq output
5. output validation
6. state update events
7. uncertainty handling
8. resolution confirmation

PHASE 3 — MEMORY INTEGRATION

1. completed-case pipeline
2. memory candidate creation
3. identity lookup
4. reinforcement
5. contradiction update
6. deprecation
7. Qdrant write
8. prior-case retrieval

PHASE 4 — REALTIME VOICE

1. LiveKit
2. Flux
3. partials
4. stabilizer
5. prefetch
6. EOT
7. Groq streaming
8. Rime streaming
9. interruptions

PHASE 5 — END-TO-END HARDENING

1. concurrency tests
2. stale-generation tests
3. network failure tests
4. provider timeout tests
5. cancellation tests
6. long session tests
7. latency benchmarks
8. memory consistency
9. production observability
10. deployment.


# REFERENCE END-TO-END PSEUDOCODE

async def handle_turn(turn):

    partials = turn.transcript_stream

    async for partial in partials:

        stable_query = stabilizer.update(
            partial
        )

        if not intent_router.is_technical(
            stable_query
        ):
            continue

        entities = extractor.extract(
            stable_query
        )

        await retrieval.prefetch(
            stable_query,
            **entities,
        )

    final_text = await turn.final_transcript()

    intent = intent_router.classify(
        final_text
    )

    if intent.is_general:

        await llm.direct_response(
            final_text
        )

        return

    observation = observation_extractor.extract(
        final_text
    )

    state_engine.apply(
        observation.to_events()
    )

    evidence = await retrieval.retrieve(
        final_text,
        state=state_engine.state,
    )

    context = context_engine.build(
        state=state_engine.state,
        evidence=evidence,
    )

    decision = await diagnosis.decide(
        state=state_engine.state,
        context=context,
    )

    validated = decision_validator.validate(
        decision
    )

    state_engine.apply(
        validated.events
    )

    await voice.speak_stream(
        validated.response
    )


# DESIGN EXAMPLE — WINDOWS WIFI CASE

Example diagnostic session:

Technician:
"My Dell Latitude keeps losing Wi-Fi."

State should capture:
- OEM = Dell
- model = Latitude (exact model if later known)
- symptom = Wi-Fi disconnects
- category = networking

Speculative retrieval can begin while the technician continues.

Technician:
"It mostly happens after waking from sleep."

This is a meaningful new observation.

Query stabilizer should recognize:
- sleep/resume context is new
- retrieval query should be updated

Context may retrieve:
- Dell/Windows sleep-related Wi-Fi driver cases
- adapter power-management procedures
- prior confirmed cases

Reasoning may form hypotheses:
A. adapter driver
B. power management
C. DHCP/network stack
D. access point

If another Wi-Fi network works:
that is supporting evidence against an access-point-global failure.

The next action should be selected based on information value, not generic troubleshooting boilerplate.


# DESIGN EXAMPLE — HARDWARE THERMAL CASE

Example:

Technician:
"XJ-420 is overheating and showing ERR-17."

The current project tests use XJ-420 as synthetic diagnostic equipment to validate the brain architecture. This example is not part of the actual Windows domain and should not be interpreted as expanding product scope.

The test demonstrates the architecture:

fault code
+
equipment context
+
symptom
→
hybrid retrieval
→
case/resolution memory
→
grounded reasoning

The same architecture will later be applied to actual Windows/OEM diagnostic entities.


# DESIGN EXAMPLE — MEMORY EVOLUTION

Suppose three confirmed Dell laptop cases show:

Case 1:
Wi-Fi drops after sleep → driver update resolves.

Case 2:
Wi-Fi drops after sleep → reinstalling adapter driver resolves.

Case 3:
Wi-Fi drops after sleep → power-management setting resolves.

The memory system should not blindly conclude:
"Wi-Fi after sleep always means driver."

Instead it may maintain:

Pattern:
Wi-Fi failures after sleep can involve adapter driver/power management.

Evidence:
3 confirmed cases.

Specific resolution memories:
- driver update
- driver reinstall
- power-management change

Contradiction/alternative:
power-management cause confirmed in one case.

Future retrieval can surface the pattern plus specific successful cases.


# DESIGN EXAMPLE — CONTRADICTION

Suppose:

Memory A:
"Windows 11 Wi-Fi issue after sleep was resolved by disabling adapter power management."

Memory B:
"Same symptom on another model was resolved by installing a newer OEM driver."

The system should preserve both.

If the current equipment model matches B:
B should receive stronger equipment-specific relevance.

If the model is unknown:
both may be useful.

The LLM should receive enough evidence to avoid pretending there is one universal cause.


# DESIGN EXAMPLE — INTERRUPTED RESPONSE

Technician asks:
"Why is my Wi-Fi dropping?"

Groq starts generating.

Rime begins speaking:
"It could be related to..."

Technician interrupts:
"Actually, Ethernet doesn't work either."

Required behavior:

- Rime stops.
- Current generation becomes stale.
- Groq response is cancelled if possible.
- New observation is processed.
- State now includes Ethernet failure.
- Query stabilizer updates.
- Retrieval speculation may start for broader networking issue.
- New reasoning generation starts.

The old Wi-Fi-only response must not resume.


# DESIGN EXAMPLE — GENERAL REQUEST

Technician:
"Thanks, that's clear."

Do not:
- query Qdrant
- run hybrid retrieval
- create memory
- modify diagnostic state

Respond directly and quickly.

This is why a cheap intent/router layer is useful.


# DESIGN EXAMPLE — DEAD AIR FALLBACK

Technician asks a technical question.

Normally:

STT
→ retrieval
→ Groq
→ Rime

If an exceptional delay occurs, a very short hardcoded Rime filler may be used.

Example concept:
"One moment."

But this should not happen routinely.

The preferred solution is to reduce actual latency through:

- speculation
- caching
- streaming
- routing
- context efficiency
- connection reuse
- bounded concurrency

Filler is a last-resort perceived-latency patch, not an architecture.


# REPOSITORY QUALITY BAR

A component is not "done" merely because one happy-path test passes.

For retrieval:
- happy path
- timeout
- stale prefetch
- failed prefetch
- concurrent prefetch
- duplicate prefetch
- cache miss
- cache hit
- filter correctness
- cancellation

For memory:
- creation
- reinforcement
- contradiction
- deprecation
- identity
- provenance
- concurrent update semantics

For state:
- invalid event
- duplicate
- collision
- stale
- rollback
- recovery

For voice:
- partials
- final turn
- interruption
- cancellation
- stale response
- reconnect
- provider failure

For LLM:
- valid structured output
- malformed output
- timeout
- cancellation
- refusal/uncertainty
- unsupported claim
- state update validation


# DO NOT OVERENGINEER

The project is ambitious, but unnecessary complexity is still bad.

Do not create:
- five wrappers around one function
- generic framework abstractions before use
- speculative microservices
- needless queues
- complicated distributed state
- premature caching layers
- custom vector algorithms when Qdrant already provides them

Use the simplest architecture that preserves the invariants.

Complexity is justified when it directly improves:
- correctness
- latency
- memory quality
- concurrency safety
- maintainability
- observability.


# IMPLEMENTATION STYLE

Prefer:

- typed Python
- dataclasses or explicit models where appropriate
- async I/O for external providers
- small deterministic pure functions for planning/normalization
- explicit exceptions for invalid state transitions
- immutable/frozen structures where useful
- dependency injection for repositories/providers
- testable interfaces
- clear names
- structured logging
- explicit timeouts
- explicit cancellation

Avoid:
- global mutable state
- hidden network calls
- blocking I/O in async paths
- swallowing exceptions silently
- magic constants without explanation
- implicit state mutation
- giant functions with unrelated responsibilities.


# CURRENT ARCHITECTURAL NORTH STAR

The final FieldMate brain should look conceptually like:

                    CURRENT CASE
                         │
                         ▼
                  DIAGNOSTIC STATE
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
       observations   hypotheses   completed tests
            │            │            │
            └────────────┼────────────┘
                         ▼
                RETRIEVAL ORCHESTRATOR
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           dense       sparse      hybrid
             └───────────┼───────────┘
                         ▼
                       QDRANT
                         │
                         ▼
                  MEMORY / EVIDENCE
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          supporting            conflicting
           evidence              evidence
              └──────────┬──────────┘
                         ▼
                 CONTEXT INTELLIGENCE
                         │
                         ▼
                  GROQ REASONING
                         │
                         ▼
                  VALIDATED DECISION
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          state events          response text
              │                     │
              ▼                     ▼
        STATE ENGINE              RIME
              │                     │
              └──────────┬──────────┘
                         ▼
                    TECHNICIAN

After a successful case:

diagnostic state
    ↓
case extraction
    ↓
memory evolution
    ↓
Qdrant
    ↓
future cases benefit

That feedback loop is the heart of FieldMate.


# FINAL INSTRUCTIONS TO THE CODING AGENT

You are working on FieldMate, not a generic demo.

Read the repository.

Read this file.

Read the deeper architecture/vendor documents when present.

Preserve the existing working components.

The most important things are:

1. Diagnostic state is authoritative.
2. Events are immutable/idempotent/atomic.
3. Memory evolves from evidence.
4. Contradictions are preserved.
5. Case extraction is conservative.
6. Memory identity is deterministic.
7. Qdrant is the real domain-memory substrate.
8. Dense/sparse/hybrid retrieval is adaptive.
9. Speculative retrieval is allowed to run ahead of the user.
10. Speculative retrieval must never block the user.
11. Partial STT must be stabilized before launching searches.
12. Non-technical input can bypass retrieval.
13. Context Intelligence decides what evidence the LLM actually sees.
14. Groq reasons over state/evidence; it does not own state.
15. Rime streams and can be interrupted.
16. Stale asynchronous work must never overwrite newer turns.
17. Optimize perceived latency, not merely individual API latency.
18. Preserve tests and add stronger tests as the system grows.
19. Use official current documentation for vendor APIs.
20. Do not invent behavior.
21. Do not put secrets into the repository.
22. Do not silently broaden the Windows/OEM scope.
23. Build the complete system in coherent vertical slices.

The objective is not to make the code look sophisticated.

The objective is to make FieldMate a reliable, low-latency, evidence-grounded diagnostic partner that remembers what happened and gets better from confirmed outcomes.
