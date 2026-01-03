Explore geometry of faithfulness (vibe coded with Gemini through Google AI Studio) using Gemma models. Executive summary can be found here: [Google doc](https://docs.google.com/document/d/1OckbA8KBIY7NwvNphmEqetDQVHjinojLluiA9SoP8YQ/edit?usp=sharing)

Question:

We investigated if LLM faithfulness relies on a steerable linear "suppression switch," enabling mathematical control over hallucinations and context adherence at inference time without changing prompts.


Conclusion:

We identified faithfulness as a steerable, linear "suppression switch" for internal memory. This mechanism is universal across declarative domains, enabling precise control over context adherence. However, it does not govern procedural reasoning, suggesting logic relies on distinct circuits separate from factual retrieval.


Technical Setup:

We quantified the Faithfulness-Truthfulness trade-off by measuring logit shifts between context-compliant (e.g., "Green") and memory-based (e.g., "Blue") tokens. Using gemma-3-270m-it, we constructed a "Knowledge Conflicts" dataset, extracting vectors from Geography/Science facts and testing on held-out domains like Translation. Our methodology employed Mass-Mean Difference to isolate a faithfulness vector (V_{faith}) by averaging the difference between faithful and truthful activations. We then applied Negative Steering (-coefficient* V_{faith}) at Layer 10 to identify the steering coefficient required to flip the model's priority from context adherence to internal truth.


Limitation:

Key limitations include the small model scale (270M), small dataset, and a narrow focus on arithmetic. Future research should replicate these experiments on larger models to test scaling laws (e.g. steering coefficient vs model size) and extract specific procedural vectors, addressing the inability of declarative faithfulness to govern logical reasoning tasks.


