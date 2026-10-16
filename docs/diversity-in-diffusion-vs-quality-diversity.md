# Is "diversity in diffusion models" quality-diversity, or does it need reinforcement learning? (research note, 2026-10-16)

**Question (owner).** The 2026-10-14 sweep and the 2026-10-15 runs tested methods that make a diffusion
model's samples more diverse. Are those methods a form of quality-diversity optimisation, or do they need
special reinforcement-learning algorithms? Which family does the team proposer belong to?

**Short answer.** They are three different things that share one word.
1. *Quality-diversity* is a search problem: fill a user-chosen descriptor space with the best solution per
   cell. Its object is an **archive**.
2. *Diversity in diffusion sampling* is a property of a **sample set**: how spread out the outputs of one
   generator are, measured without a descriptor (recall, Vendi score, pairwise distance). Most methods
   change the sampler or the training objective and need no reinforcement learning.
3. *Reinforcement-learning fine-tuning with a diversity term* is one **mechanism** inside family 2, used
   when the model is being optimised for a reward and would otherwise collapse.

The team proposer is a quality-diversity problem (descriptor = the six-species line-up, quality = win rate
against the top-50 meta) whose **emitter** is a diffusion model. The diffusion-diversity methods are tools
for the emitter; the archive and the elite rule are where the problem's diversity actually lives.

## Primary sources, by family

### Quality-diversity (archive-based search)
- 🟢 Mouret & Clune, *Illuminating search spaces by mapping elites*, arXiv 2015 —
  [alphaXiv 1504.04909](https://www.alphaxiv.org/abs/1504.04909). MAP-Elites keeps "a map of high-performing
  solutions at each point in a space defined by dimensions of variation that a user gets to choose."
  Diversity is defined by the user's descriptor axes, never by the data.
- 🟢 Pugh, Soros & Stanley, *Quality Diversity: A New Frontier for Evolutionary Computation*, Frontiers in
  Robotics and AI 2016 — [doi 10.3389/frobt.2016.00040](https://www.frontiersin.org/articles/10.3389/frobt.2016.00040).
  Names the class: algorithms that "return an archive of diverse, high-quality behaviors in a single run";
  the descriptor is called a behaviour characterisation.
- 🟢 Ding, Zhang, Clune, Spector & Lehman, *Quality Diversity through Human Feedback*, ICML 2024 —
  [alphaXiv 2310.12103](https://www.alphaxiv.org/abs/2310.12103). Learns the descriptor from human similarity
  judgements and applies it to a text-to-image diffusion model's latent space: the diffusion model is the
  thing being searched over, the archive is still the object.
- 🟢 Hegde, Batra, Zentner & Sukhatme, *Generating Behaviorally Diverse Policies with Latent Diffusion
  Models*, NeurIPS 2023 — [alphaXiv 2305.18738](https://www.alphaxiv.org/abs/2305.18738). The reverse
  direction: a MAP-Elites archive of policies is distilled into one diffusion model conditioned on the
  behaviour measures (13× compression, 89% coverage recovered). The archive comes first; diffusion stores it.

### Diversity of a diffusion model's samples, no archive, no reinforcement learning
- 🟢 Sadat et al., *CADS*, ICLR 2024 — [alphaXiv 2310.17347](https://www.alphaxiv.org/abs/2310.17347).
  Annealed noise on the conditioning embedding; recall 0.32 → 0.62 on class-conditional ImageNet at guidance 5.
- 🟢 Kynkäänniemi et al., *Applying Guidance in a Limited Interval*, NeurIPS 2024 —
  [alphaXiv 2404.07724](https://www.alphaxiv.org/abs/2404.07724). Guidance only in the middle noise levels;
  the gain is "primarily by improving Recall".
- 🟢 Corso et al., *Particle Guidance*, ICLR 2024 — [alphaXiv 2310.13102](https://www.alphaxiv.org/abs/2310.13102).
  A joint repulsive potential couples the samples of one batch; diversity is measured as in-batch
  dissimilarity and recall, with no descriptor.
- 🟢 Morshed & Boddeti, *DiverseFlow*, CVPR 2025 — [alphaXiv 2504.07894](https://www.alphaxiv.org/abs/2504.07894).
  A determinantal point process inside the flow ODE; the kernel is a learned feature similarity, not a
  descriptor grid.
- 🟢 Lamont et al., *Free Lunch for Pass@k?*, arXiv preprint 2026 — [alphaXiv 2603.04893](https://www.alphaxiv.org/abs/2603.04893);
  🟢 Lys et al., *D5P4*, arXiv preprint 2026 — [alphaXiv 2603.19146](https://www.alphaxiv.org/abs/2603.19146).
  The two masked-diffusion batch-coupling methods; both training-free.
- 🟢 Friedman & Dieng, *The Vendi Score*, TMLR 2023 — [alphaXiv 2210.02410](https://www.alphaxiv.org/abs/2210.02410).
  The metric this family reports: exponential of the entropy of a similarity matrix's eigenvalues, "does not
  require a reference dataset". A kernel replaces the descriptor.

### Reinforcement-learning fine-tuning with a diversity term (one mechanism inside the family above)
- 🟢 Cideron et al., *Diversity-Rewarded CFG Distillation*, ICLR 2025 —
  [alphaXiv 2410.06084](https://www.alphaxiv.org/abs/2410.06084). REINFORCE with a pairwise embedding-distance
  reward on a discrete-token music generator, then weight interpolation between a quality and a diversity
  checkpoint. Reinforcement learning is used because the model is being pushed toward a reward and would
  otherwise lose diversity; the paper's own limitation is that the diversity metric "can be hacked".
- 🟢 Uehara et al., *Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control*,
  arXiv 2024 — [alphaXiv 2402.15194](https://www.alphaxiv.org/abs/2402.15194). Entropy regularisation
  against the pretrained model to keep reward fine-tuning from collapsing.
- 🟢 Farnia, Jalali & Ospanov, *Exposing Diversity Bias in Deep Generative Models*, ICLR 2026 —
  [alphaXiv 2602.14682](https://www.alphaxiv.org/abs/2602.14682). The theory for why any generator trained on
  a finite sample loses entropy, with a training-time entropy bonus as one correction; no reinforcement
  learning involved.

## What distinguishes the three (the test to apply to any paper)
| | descriptor chosen by the user | object returned | needs a reward signal | needs reinforcement learning |
|---|---|---|---|---|
| quality-diversity | yes (behaviour characterisation) | an archive | yes (quality per cell) | no |
| diffusion-sample diversity | no (a kernel or recall) | a sample set | no | no |
| diversity-rewarded fine-tuning | no | a model | yes | yes (or a differentiable reward) |

## Where the team proposer sits
- **The problem is quality-diversity.** The six-species line-up is a natural descriptor (the project's
  "winning line-ups" count = occupied cells with quality ≥ 0.531), win rate is the quality, and the loop
  already keeps elites. The deep-surrogate MAP-Elites run (`dsame-experiment.md`, 2026-10-01) was this family
  made explicit and beat the loop on fresh battles (+0.084 [0.044, 0.123]) while concentrating compositions.
- **The emitter is a diffusion model.** Everything tried on 2026-10-14/15 (guidance interval, CADS, species
  repulsion) belongs to the sample-diversity family and widens the emitter's output. All three failed the
  −0.03 win-rate bound as one-shot samplers; the owner's 2026-10-16 position is that a lower batch mean is
  acceptable when the search explores different Pokémon, judged on best fresh team and off-core rosters.
- **No special reinforcement-learning algorithm is required.** The loop's elite re-steer is already a
  reward-weighted fine-tune (the closest relative is Cideron et al.'s diversity-rewarded distillation). Adding
  a pairwise species-distance reward to it is the one reinforcement-learning-shaped idea not yet tried; the
  entropy and redundancy terms tried in *selection* (2026-09-02, 2026-10-04) all failed, so expect little.
- **Diffusion-BBO does not change this.** Its learning rule was ported into the loop on 2026-10-16
  (`cem-dbbo.md`): all-data fine-tuning and a delivered-based ask were null; the only near-signal was the
  sampling change (taboo + sets-only temperature, +0.035 [−0.002, +0.072]). Diffusion-BBO conditions the
  same way this project does; the difference is the loop, not the model.

## What to say about this approach (talking points)
1. Diversity has to be *defined* before it can be optimised: descriptor (quality-diversity), kernel (Vendi,
   determinantal point process) or recall. The project uses a descriptor, which makes it quality-diversity.
2. Sampling-time diversity methods are cheap and training-free, and in this project they move line-ups by
   about the per-cell noise because the model holds only ~50 species it ever picks; the ceiling is the
   emitter's support, not the sampler.
3. The lever that worked is archive-side: taboo on compositions (~440 line-ups per 512) and the surrogate
   MAP-Elites archive. That is the quality-diversity half of the story.
4. Reinforcement learning enters only if the model itself is optimised for the reward; the loop's re-steer
   already does that, so the question is which regulariser, not which algorithm.
