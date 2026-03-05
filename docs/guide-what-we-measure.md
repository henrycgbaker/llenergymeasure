# What We Measure and Why It Matters

This guide explains what llenergymeasure measures — energy, throughput, and FLOPs — in plain language. No programming knowledge required.

---

## Why AI Energy Measurement Matters

Artificial intelligence models run on computers. Those computers consume electricity. As AI becomes more widely used — in search, in healthcare, in government services, in consumer products — the energy cost of running these systems has become a significant policy concern.

Data centres that host AI infrastructure already account for a growing share of global electricity demand. Training a large AI model once can consume as much electricity as a transatlantic flight. But training happens once; *inference* — using a trained model to generate answers — happens millions or billions of times per day across deployed systems. That repeated cost is what llenergymeasure measures.

Understanding the energy cost of AI inference enables:

- **Informed procurement decisions** — choosing AI systems that deliver equivalent capability with lower energy use
- **Sustainability accounting** — including AI infrastructure in carbon footprint reporting
- **Policy design** — setting energy efficiency standards or disclosure requirements for AI systems
- **Comparative research** — evaluating whether a more capable AI system is worth its higher energy cost

llenergymeasure makes this measurement rigorous, reproducible, and comparable across different models and deployment configurations.

---

## Energy (Joules): How Much Electricity Does One Inference Use?

**What it is:** The amount of electrical energy consumed by the computer's graphics processor (GPU) to complete one AI inference — that is, to process an input and generate a response.

**Analogy:** Think of it like measuring fuel consumption for a car journey. Just as you might ask "how many litres of fuel does this car use for a 10-kilometre drive?", we ask "how many joules of energy does this AI model use to process 100 prompts?"

**The unit:** Joules. One joule is one watt of power used for one second. A GPU might draw 300 watts of power; if an inference takes 2 seconds, that is 600 joules. For context, a typical smartphone battery stores roughly 15,000 joules.

**What llenergymeasure reports:**

- **Total energy** — the raw GPU energy consumed during the experiment
- **Baseline power** — the idle power the GPU draws even when doing nothing (like a car engine idling)
- **Adjusted energy** — total energy minus the baseline, representing energy specifically attributable to the inference work

The adjusted figure is the most meaningful for comparison, because it isolates the cost of the AI work rather than the general cost of keeping the hardware running.

---

## Throughput (Tokens per Second): How Fast Does the AI Produce Output?

**What it is:** How many units of output (called "tokens") the AI model produces per second.

**What is a token?** In AI language models, text is broken into small chunks called tokens. A token is roughly equivalent to a short word or part of a word. "Hello, world!" is about 4 tokens. This is the unit AI models work with internally.

**Analogy:** Think of it like typing speed — words per minute. A faster typist produces more words in the same time. A model with higher throughput produces more tokens in the same time.

**Why it matters:** Throughput affects how quickly users get responses, and how many users a given system can serve simultaneously. A faster model can handle more traffic with the same hardware.

**The trade-off:** A model that runs more slowly might actually be more *energy efficient per unit of output* — it might use less total energy to produce the same amount of text. Throughput alone does not tell you about energy efficiency; you need to combine it with energy measurement.

---

## FLOPs (Floating Point Operations): How Much Computational Work Does the Model Do?

**What it is:** A count of the number of mathematical calculations the AI model performs during inference.

**Analogy:** Think of it like counting the number of individual calculations a student makes when solving a maths problem. A more complex problem requires more calculations. Similarly, a larger AI model performs more calculations to generate each response.

**The unit:** FLOPs — "Floating Point Operations". Larger models perform billions or trillions of FLOPs per inference (reported as GFLOPs or TFLOPs respectively).

**Why it matters:** FLOPs provide a hardware-independent measure of computational complexity. This is useful for:

- Comparing models of different sizes fairly (a larger model naturally uses more energy partly because it does more work)
- Estimating theoretical efficiency limits (a model using many FLOPs but little energy is running efficiently; a model using few FLOPs but high energy suggests hardware or configuration inefficiency)
- Normalising comparisons across different GPU hardware

FLOPs are estimated by llenergymeasure based on the model architecture. They cannot be measured directly during inference but can be calculated from model properties.

---

## Why These Three Metrics Together?

Each metric alone can mislead:

- **Energy alone** does not account for how much useful work was done. A model that uses twice the energy might produce ten times the output.
- **Throughput alone** does not capture efficiency. A fast but power-hungry model may cost more to run than a slower, more efficient one.
- **FLOPs alone** describe complexity but not actual hardware utilisation or energy draw.

Together, the three metrics enable meaningful comparisons:

| Question | Metrics needed |
|----------|---------------|
| Which model costs more to run? | Energy per inference |
| Which model is faster per user? | Throughput (tokens/second) |
| Which model is most efficient per unit of output? | Energy ÷ throughput (joules per token) |
| Is this hardware being used efficiently? | FLOPs vs energy (FLOPs/joule) |
| Is this model worth its higher energy cost? | All three, combined with accuracy |

The goal of llenergymeasure is to make these comparisons rigorous and reproducible — giving policy makers and researchers the data they need to evaluate AI systems beyond their headline capabilities.

---

## Further Reading

- [How to Read llenergymeasure Output](guide-interpreting-results.md) — what the numbers mean in practice
- [Running Your First Measurement](guide-getting-started.md) — a step-by-step guide for running your first measurement
- [Comparison with Other Benchmarks](guide-comparison-context.md) — how llenergymeasure relates to MLPerf, AI Energy Score, and other tools
