"""Phase two: optimize the DSPy-structured completion program, then test it unchanged.

See docs/dspy-structure-experiment.md. Optimization uses freshly sampled starters
(never the 20 test starters or a meta-pool team), the same model, provider,
reasoning level, sampling settings, prompt version and opponent pool as a
finished baseline run. The result is a frozen program file of instructions
and demonstrations. Its test is a normal phase-one study:

    dspy_structure_experiment.py prepare --program <output>/program.json ...

so the optimized arm meets the identical request, validation and battle path.

prepare makes no API calls. optimize makes paid calls through DSPy's LM client.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import dspy

import dspy_structure as S
import llm_baseline as B

SPLITS = ('train', 'dev')


def prepare(args):
    """Sample optimization starters disjoint from the test starters and the pool."""
    import run_reasoning_pilot as P
    parent = B.load_manifest(SimpleNamespace(output=args.parent))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty')
    if parent.get('prompt_version') == 'mb-grounded-v2':
        raise ValueError('mb-grounded-v2 varies its input fields by task; optimize v1 or v3 prompts')
    count = args.train_starters + args.dev_starters
    m = P.fresh_starters(parent, count, args.seed)  # excludes parent starters and every pool team
    split = {str(i): 'train' if i < args.train_starters else 'dev' for i in range(count)}
    m.update(optimization=dict(parent=str(args.parent), parent_manifest_sha256=B.digest(parent),
                               split=split, adapter=args.adapter,
                               rule='largest mask per family; starters sampled fresh'))
    constraints = getattr(args, 'constraints', None)
    if constraints:
        m['optimization']['constraints'] = constraints
    m['structure'] = S.config(args.adapter, constraints=constraints)
    for trial in m['trials']:
        S.check_information(m, trial)
    B.save(args.output / 'manifest.json', m)
    print(json.dumps(dict(output=str(args.output), train=args.train_starters, dev=args.dev_starters,
                          tasks=len(m['trials']))))


class Refused(str):
    """An answer the legal-choice type refused; its text is the reason."""


def lenient(adapter):
    """The same adapter, made to suit optimization with legal-choice types.

    - A reply refused by the answer type is returned as `Refused` instead of
      raising, so the metric can give GEPA the exact reason (as it does for
      validator messages). Raising would leave GEPA only a generic parse note.
    - No `response_format`: the phase-one requests GEPA's result is tested with
      never send one, so the optimized requests must not either.
    """
    def parse(self, signature, completion):
        try:
            return type(adapter).parse(self, signature, completion)
        except dspy.AdapterParseError as exc:
            return {name: Refused(S.refusal_message(exc)) for name in signature.output_fields}

    cls = type('Lenient' + type(adapter).__name__, (type(adapter),),
               dict(parse=parse, _prepare_response_format=lambda self, *a, **k: None))
    wrapped = copy.copy(adapter)
    wrapped.__class__ = cls
    return wrapped


class Completion(dspy.Module):
    """The phase-one request as a DSPy program: one Predict, no added reasoning field.

    With legal-choice types each task has its own answer type, but GEPA rewrites
    one predictor's instructions. So the predictor keeps a generic signature
    whose instructions GEPA edits, and each call sends that task's typed
    signature carrying the current instructions.
    """

    def __init__(self, manifest):
        super().__init__()
        self.manifest = manifest
        self.trials = {t['id']: t for t in manifest['trials']}
        generic = dict(manifest, structure={k: v for k, v in manifest['structure'].items() if k != 'constraints'})
        sig, _, _ = S.parts(generic, manifest['trials'][0])
        self.predict = dspy.Predict(sig)

    def forward(self, trial_id=None, **inputs):
        if not self.manifest['structure'].get('constraints'):
            return self.predict(**inputs)
        typed, _, _ = S.parts(self.manifest, self.trials[trial_id])
        return self.predict(signature=typed.with_instructions(self.predict.signature.instructions), **inputs)


def examples(manifest, split):
    keys = None
    rows = []
    for trial in manifest['trials']:
        if manifest['optimization']['split'][str(trial['start'])] != split:
            continue
        _, inputs = S.baseline_parts(manifest, trial)
        if keys is None:
            keys = list(inputs)
        elif list(inputs) != keys:
            raise ValueError('input fields differ between tasks')
        rows.append(dspy.Example(**inputs, trial_id=trial['id']).with_inputs(*keys, 'trial_id'))
    return rows


class Metric:
    """Legality through the unchanged checks; optionally the 50-battle panel.

    Returns a score with written feedback, so GEPA can learn from the exact
    validator message. Score 0 when illegal; legal answers score `alpha`, plus
    (1 - alpha) × win rate when battles are enabled.
    """

    def __init__(self, manifest, output, battles=False, alpha=.5, validator=None, h=None, battle=None):
        self.manifest, self.output, self.battles, self.alpha = manifest, Path(output), battles, alpha
        self.trials = {t['id']: t for t in manifest['trials']}
        self.validator, self.h = validator, h
        self.battle = battle or self.battle_panel
        self.cache = {}

    def check(self, trial, pred):
        if isinstance(pred.fills, Refused):
            raise ValueError(str(pred.fills))
        if self.manifest['structure'].get('constraints'):
            answer = S.as_fills(pred.fills, trial['paths'])
        else:
            answer = {'fills': [f.model_dump() for f in pred.fills]}
        team = B.apply_completion(self.manifest['starts'][trial['start']], trial['paths'], answer)
        error = self.validator(B.paste(team, self.h))
        if error:
            raise ValueError(error)
        return team

    def battle_panel(self, trial, team):
        label = f"llm-{trial['id']}-{B.digest(team)[:12]}"
        result = {}
        B.battle(SimpleNamespace(output=self.output), candidates_override=[(label, team)],
                 on_result=lambda _, r: result.update(r))
        return result['wins'] / result['battles']

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        trial = self.trials[gold.trial_id]
        try:
            team = self.check(trial, pred)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            return dspy.Prediction(score=0.0, feedback=f'Rejected: {exc}')
        if not self.battles:
            return dspy.Prediction(score=1.0, feedback='Legal: passed the shape checks and the Showdown validator.')
        key = B.digest(dict(team=team, start=trial['start']))
        if key not in self.cache:
            self.cache[key] = self.battle(trial, team)
        win = self.cache[key]
        return dspy.Prediction(score=self.alpha + (1 - self.alpha) * win,
                               feedback=f'Legal. Won {round(50 * win)} of 50 battles against the opponent pool.')


def lm_for(manifest, model=None, **overrides):
    """Same sampling settings and provider route as the baseline request."""
    import os
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY is not configured; no requests sent')
    settings = dict(temperature=manifest['temperature'], max_tokens=manifest['max_tokens'], cache=False,
                    extra_body=dict(reasoning=manifest.get('reasoning', {'enabled': True}),
                                    provider=manifest.get('provider', {'order': ['novita'], 'allow_fallbacks': False})))
    settings.update(overrides)
    return dspy.LM('openrouter/' + (model or manifest['model']), api_key=key, **settings)


ANSWER_CONTRACT = (
    'The answer format is enforced by the program and cannot be changed by the instruction: each example '
    'shows the exact answer shape its task requires. Never tell the assistant to drop, rename or add answer '
    'fields. In particular every Stat Points object (an `evs` value) must keep its `total` field, written as '
    'the sum of HP, Atk, Def, SpA, SpD and Spe, at most 66. The instruction may explain how to choose good, '
    'legal values; it must not contradict the required shape.')


def compact_schema(schema, limit=8):
    """A JSON schema with long value lists replaced by their size: the shape, not hundreds of names."""
    if isinstance(schema, dict):
        out = {k: compact_schema(v, limit) for k, v in schema.items() if k != 'enum'}
        if 'enum' in schema:
            values = schema['enum']
            out['enum' if len(values) <= limit else 'allowed_values'] = (
                values if len(values) <= limit else f'one of {len(values)} values listed in the task schema')
        return out
    if isinstance(schema, list):
        return [compact_schema(v, limit) for v in schema]
    return schema


class ShapeAwareProposer(dspy.teleprompt.gepa.instruction_proposal.InstructionProposer):
    """GEPA's built-in proposer, told the answer shape each task requires.

    The reflection model otherwise sees only the instructions and example
    answers, never the typed answer schema, so it can write rules that
    contradict it (it once wrote "do not add a total field"). Each reflection
    example gets its task's own answer shape, and a standing note says the
    shape is fixed by the program.
    """

    def __init__(self, **kwargs):
        super().__init__(additional_instructions=ANSWER_CONTRACT, **kwargs)

    def _prepare_examples(self, examples):
        prepared = super()._prepare_examples(examples)
        for example in prepared:
            inputs = example.get('Inputs') or {}
            if 'masked_team' in inputs and 'requested_paths' in inputs:
                answer = S.legal_completion_type(inputs['masked_team'], inputs['requested_paths'])
                example['Required answer shape'] = compact_schema(answer.model_json_schema())
        return prepared


def reflection_lm_for(model, effort='high', num_retries=3):
    """GEPA's prompt writer, on its own route.

    It never answers a task, so it keeps none of the task model's request
    settings: no provider pin (the task's single-provider route refuses other
    vendors' models), its own reasoning effort instead of the task's, and no
    temperature (reasoning models do not accept one).
    """
    import os
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY is not configured; no requests sent')
    return dspy.LM('openrouter/' + model, api_key=key, max_tokens=32000, cache=False, num_retries=num_retries,
                   extra_body=dict(reasoning={'effort': effort}))


def export(program, manifest, optimizer):
    """Frozen instructions and demonstrations in the phase-one program format."""
    keys = list(program.predict.signature.input_fields)
    demos = []
    for demo in program.predict.demos:
        demo = dict(demo)
        fills = demo['fills']
        demos.append({**{k: demo[k] for k in keys},
                      'fills': [f.model_dump() if hasattr(f, 'model_dump') else f for f in fills]})
    return dict(instructions=program.predict.signature.instructions, demos=demos, optimizer=optimizer)


def optimize(args):
    m = B.load_manifest(SimpleNamespace(output=args.output))
    if 'optimization' not in m:
        raise ValueError('run prepare first')
    train, dev = examples(m, 'train'), examples(m, 'dev')
    h, _ = B.runtime()
    validator = h.Validator()
    # Retries go to LiteLLM, which backs off exponentially; more of them ride out shared-pool rate limits.
    lm = lm_for(m, num_retries=args.num_retries)
    adapter = S.adapter_for(m['structure'])
    if m['structure'].get('constraints'):
        if args.optimizer != 'gepa':
            raise ValueError('MIPROv2 demonstrations cannot carry per-task legal-choice types; use GEPA')
        adapter = lenient(adapter)
    try:
        dspy.configure(lm=lm, adapter=adapter)
        metric = Metric(m, args.output, battles=args.battles, alpha=args.alpha, validator=validator, h=h)
        program = Completion(m)
        if args.optimizer == 'gepa':
            reflection = reflection_lm_for(args.reflection_model, args.reflection_effort, args.num_retries) \
                if args.reflection_model else lm_for(m, num_retries=args.num_retries)
            proposer = ShapeAwareProposer() if m['structure'].get('constraints') else None
            optimizer = dspy.GEPA(metric=metric, auto=args.budget, reflection_lm=reflection,
                                  instruction_proposer=proposer,
                                  add_format_failure_as_feedback=True, num_threads=args.threads,
                                  seed=args.seed, log_dir=str(args.output / 'gepa'), track_stats=True)
        else:
            optimizer = dspy.MIPROv2(metric=lambda g, p, t=None: metric(g, p).score, auto=args.budget,
                                     max_bootstrapped_demos=args.max_demos, max_labeled_demos=0,
                                     num_threads=args.threads, seed=args.seed,
                                     log_dir=str(args.output / 'mipro'))
        before = dspy.Evaluate(devset=dev, metric=lambda g, p, t=None: metric(g, p).score,
                               num_threads=args.threads)(program).score
        optimized = optimizer.compile(program, trainset=train, valset=dev)
        after = dspy.Evaluate(devset=dev, metric=lambda g, p, t=None: metric(g, p).score,
                              num_threads=args.threads)(optimized).score
        cost = sum(entry.get('cost') or 0 for entry in lm.history)
        reflection_cost = None
        if args.optimizer == 'gepa' and args.reflection_model:
            reflection_cost = sum(entry.get('cost') or 0 for entry in reflection.history)
        settings = dict(name=args.optimizer, budget=args.budget, battles=args.battles, alpha=args.alpha,
                        threads=args.threads, num_retries=args.num_retries,
                        constraints=m['structure'].get('constraints'),
                        reflection_model=args.reflection_model,
                        reflection_effort=args.reflection_effort if args.reflection_model else None,
                        reflection_lm_cost_usd=reflection_cost, max_demos=args.max_demos, seed=args.seed,
                        train_tasks=len(train), dev_tasks=len(dev), dev_score_before=before,
                        dev_score_after=after, task_lm_cost_usd=cost, dspy_version=dspy.__version__)
        B.save(args.output / 'program.json', export(optimized, m, settings))
        optimized.save(str(args.output / 'dspy-program.json'))
        print(json.dumps(settings))
    finally:
        validator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'optimize'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', type=Path, help='a baseline run folder, e.g. results/<study>/off/r0 (prepare)')
    p.add_argument('--train-starters', type=int, default=12)
    p.add_argument('--dev-starters', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--adapter', choices=S.ADAPTERS, default='chat')
    p.add_argument('--constraints', choices=S.CONSTRAINTS,
                   help='prepare: optimize with each task typed by its legal choices (dspy_structure.legal_completion_type)')
    p.add_argument('--optimizer', choices=['gepa', 'mipro'], default='gepa')
    p.add_argument('--budget', choices=['light', 'medium', 'heavy'], default='light')
    p.add_argument('--battles', action='store_true', help='score legal answers with the 50-battle panel too')
    p.add_argument('--alpha', type=float, default=.5, help='score of a legal answer before battles')
    p.add_argument('--reflection-model', help='GEPA reflection model, e.g. openai/gpt-6-luna; default is the task model')
    p.add_argument('--reflection-effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                   help='reasoning effort for --reflection-model')
    p.add_argument('--max-demos', type=int, default=0, help='MIPROv2 demonstrations (each repeats the opponent pool)')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--num-retries', type=int, default=3,
                   help='per-request retries with exponential backoff (rate limits, transient errors)')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.command == 'prepare':
        if not args.parent:
            p.error('prepare requires --parent')
        args.parent = args.parent.resolve()
        if args.train_starters < 1 or args.dev_starters < 1:
            p.error('train and dev starters must be positive')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
