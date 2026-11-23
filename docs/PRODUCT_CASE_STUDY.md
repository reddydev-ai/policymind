# PolicyMind — AI Product Case Study

**Project type:** Independent portfolio prototype · **Target audience:** AI Automation Product Manager interview panel · **Evidence boundary:** Results below refer only to this repository's implementation and tests. They are **not McKesson production results**.

## Product problem and users
Policy and operations staff lose time locating exact answers in document libraries. Wrong, stale, or uncited answers can be more damaging than a slow search. PolicyMind explores a cited-answer workflow with a clear *source-to-answer* trace.

| Persona | Need | Product requirement |
| --- | --- | --- |
| Operations analyst | Find policy language quickly | Question-based search with source and page |
| Policy owner | Control revisions | New version replaces older indexed chunks for the same source |
| Risk/compliance reviewer | Verify every answer | Evidence links and disclosed evaluation limitations |
| Engineering operator | Know whether a change is safe | CI tests, failure gates, restricted access |

## Proposed value proposition
Help a user locate policy evidence without relying on an unverifiable model assertion. The demo uses PDF ingestion, Qdrant-backed retrieval, citations, a protected FastAPI interface and a separate private evaluation API. The standalone Streamlit UI is restricted to local demo use.

**Do not infer customer adoption, time saved, or cost reductions**: these have not been measured for this independent prototype.

## Product discovery assumptions to validate
1. Users need exact page evidence more than fluent summaries.
2. A visible abstention is preferable to an unsupported answer.
3. Revised policies must supersede old index versions.
4. Reviewers need a different evidence privilege from ordinary question answering.

These are *hypotheses*, not validated customer research.

## Product metrics — definitions, not achieved KPIs

| Dimension | Proposed definition | Validation needed |
| --- | --- | --- |
| Answer coverage | Answerable questions with an evidenced response / total answerable questions | Human-labeled held-out set |
| Unsupported-answer rate | Answers with one or more material unsupported claims / all substantive answers | Claim-level manual review |
| Citation accuracy | Correct source+page citations / all inspected citations | Independent evidence labeling |
| Policy freshness | Responses referencing current active policy version / answers requiring a versioned policy | Version replacement and lifecycle test |
| User adoption | Weekly active users / eligible users | Real privacy-preserving usage telemetry |
| User value | Median task completion time compared with existing search | Controlled user study |
| Operations cost | Model + vector search + service costs per successful answer | Instrumentation and invoices |

## MVP → product roadmap
**MVP (implemented):** upload PDF, retrieve chunks, answer with citations, restrict APIs, tests for source provenance, duplicate ingestion and version replacement.

**Next, not completed:** authentic multi-user identity, per-document authorization, malicious PDF isolation, deletion/retention policies, dedicated human-labeled evaluations, trustworthy abstentions, monitoring and SLOs.

**Defer until evidence:** model fine-tuning, autonomous actions and multi-tenant deployment. Adding more AI is not inherently more valuable.

## Main product trade-offs
- Faster answers vs verifiable evidence: prioritize citations and honest uncertainty.
- Easy demo UX vs secure deployment: keep the local UI private; require identity-aware controls for production.
- Re-ingestion convenience vs document lifecycle: content-addressed IDs prevent duplicates; lifecycle semantics still need production testing.
- Quality dashboards vs true measurement: synthetic integration tests are not production model metrics.

## Demo / interview walkthrough
1. Explain policy-staleness risk and user journey.
2. Upload an **entirely synthetic** PDF in an isolated local environment.
3. Query a covered fact and inspect source/page citations.
4. Show the separately keyed evaluation endpoint and a failing access test.
5. Show a document revision and version-replacement behavior.
6. Open the CI workflow and distinguish proven checks from untested real-world quality.

## Evidence links
- [Source, implementation and setup](../README.md)
- [Security and regression workflow](../.github/workflows/security.yml)
- [Test cases](../tests/)
- [GuardEval real-service synthetic integration](https://github.com/reddydev-ai/guardeval/actions/workflows/live-integration.yml)

**Status:** Tested independent prototype, not production-certified. No company customer results or commercial impact are claimed.
