The system teaches a language model to solve math word problems more accurately.

***

Stage 1 – Value Estimation (V*)

  Uses a frozen reference model to generate multiple answers for each problem.

  Calculates an average “expected reward” based on how often the answers are correct.

  Saves this as the V* value for each problem (acts as a baseline).

Stage 2 – Policy Optimization (A-PO)*

  Trains a new model (policy) to generate better answers than the baseline.

  Compares its log-probabilities with the reference model.

  Updates itself to minimize the gap between prediction and true reward.

***

Evaluation

  Tests the trained model on the MATH-500 benchmark.

  Optionally uses a PAG-style verifier, where the model checks and revises its own answer.

***

The PAG improves reasoning by making the model act as its own verifier by generating an answer, checking it, and revising if needed.
This encourages step-by-step reasoning accuracy rather than single-shot guessing.
