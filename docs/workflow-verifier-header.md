---
created_at: 2026-09-02
updated_at: 2026-09-02
type: protocol
---

# Workflow verifier prompts — domain scoping header

Paper-reading subagents in this project keep getting stopped by the biosecurity
classifier. This is the fix: a header that goes above every verifier prompt, plus a
rule for claims that name a genomics or chemistry dataset.

## The receipt

Eight `stop_details: {type: "refusal", category: "bio"}` events across four workflow
runs — 2026-08-26 (2, `wf_c6bc4b2b-136`), 2026-08-28 (4, `wf_fea80018-5b4`),
2026-08-29 (1, `wf_dcb5a6e6-e0e`), 2026-09-02 (1, `wf_a12be98e-409`). All eight on
Fable (`claude-fable-5` ×7, `claude-fable-5-1` ×1); four attributed to alphaxiv
`answer_pdf_queries`. One refused call had 211,580 tokens of cached input and 6,523
thinking tokens before the stop landed — full PDF loaded, model deep into the method.

The cause is the literature, not the project. Discrete-diffusion *guidance* is a
genomics and drug-discovery field: [arXiv 2412.10193](https://arxiv.org/abs/2412.10193)
(Schiff et al.) evaluates on "genomic sequences, small molecule design";
[arXiv 2406.01572](https://arxiv.org/abs/2406.01572) (Nisonoff et al.) on
"small-molecules, DNA sequences and protein sequences". An agent holding a full
protein-design PDF and reasoning about steering a generative model toward a target
sequence property is textually indistinguishable from the thing the classifier
exists to catch. Nothing about VGC triggers it.

The 2026-08-26 prompt made it worse two ways: it carried no statement of what the
verification was *for*, and its claim 3 asked the agent to "quote the actual numbers"
out of the QM9 (molecules) and Species10 (genomics) tables.

## The header

Paste above `You are an adversarial verifier`:

    DOMAIN CONTEXT (this decides what you read and what you ignore):
    This verification serves a competitive Pokemon VGC team-building project — a
    search over teams encoded as 6 slots x 8 categorical fields. Nothing in this
    project designs, generates, or optimizes biological, chemical, genomic, or
    molecular sequences. The papers under review are discrete-diffusion METHOD
    papers whose benchmarks happen to be genomics and chemistry; only the method
    transfers.

    READING SCOPE:
    Read the method sections, algorithm blocks, and the specific table cells named
    in the claims. Do not read, summarize, or quote the biological, genomic,
    chemical, or molecular experimental material beyond the single metric a claim
    names. Where a claim cites a dataset from those domains, it is being used ONLY
    as a vocabulary-size or sequence-length benchmark — report the metric and the
    vocabulary size, never the molecular or biological content.

    SOURCE ORDER — stop at the first one that answers the claim:
      1. arxiv.org/abs/<id> (abstract page) and the OpenReview or PMLR page
      2. the authors' released code
      3. the full PDF, only for a claim the above cannot settle, and only the
         section that claim names
    Prefer targeted section reads over loading a whole PDF into context.

    IF A SOURCE WILL NOT OPEN — paywall, fetch failure, or a refused read — mark the
    claim UNVERIFIABLE and name the source and the reason. Do not rephrase a request
    to get a blocked source open, and do not route around it with a secondary
    summary.

## Claims naming a genomics or chemistry dataset

State what the dataset is doing in the argument. The 2026-08-26 claim 3, patched:

    3. UDLM is at parity or better than masked diffusion on small-vocabulary
       datasets (QM9 |V|=40 and Species10 |V|=12) and only loses on
       large-vocabulary natural-language data. Report the metric values and the
       vocabulary sizes from the results table. These two datasets are relevant
       here solely as small-vocabulary benchmarks — my search space is ~48
       categorical columns with small per-field vocabularies — so I need the
       numbers and |V|, nothing about what the sequences encode.

The parity-at-small-vocabulary result is the load-bearing fact for a 48-column
categorical space. Saying so converts "quote the molecule table" into a scoped
methodological question, which is what it always was.

## What this is not

Not a way around the classifier. The scoping works because the biology genuinely is
not needed — the method transfers, the benchmarks do not. If a task ever does need
those sections, the move is `/bug` from an interactive session to report the false
positive with the transcript, not a rewording. The `UNVERIFIABLE` rule above exists
so an agent surfaces a block instead of hunting for another route to the same text.
