To obtain generative probabilities (log-probabilities) from a Large Language Model using the **LightEval** framework, you need to configure your evaluation task to use the `LOGPROBS` sampling method or enable `use_logits`.

In LightEval, the output is typically structured into a `ModelResponse` object which contains a `logprobs` field. Here is how you can access and configure this.

### 1. Configure the Task Request

When defining a custom task or utilizing an existing one, the `Doc` object (which represents a single evaluation instance) must be instructed to return probabilities. You do this by setting the `sampling_methods` or `use_logits` flag.

```python
from lighteval.tasks.requests import Doc

# Example for a generative task where you want probabilities
doc = Doc(
    query="What is the capital of France?",
    choices=[],  # Empty for purely generative tasks
    instruction="Answer in one word.",
    generation_size=5,
    use_logits=True  # This tells LightEval to capture token-level information
)

```

### 2. Accessing the Probabilities in Output

LightEval's model runners (like the `AccelerateModel` or `VLLMModel`) return a `ModelResponse`. For generative tasks, the log-probabilities are stored in the `logprobs` attribute as a list of values (one for each generated token).

| Attribute | Description |
| --- | --- |
| `text` | The decoded string of the generation. |
| `logprobs` | A list of floats representing the log-probability of each generated token. |
| `input_tokens` | The IDs of the prompt tokens. |
| `output_tokens` | The IDs of the generated tokens. |

> **Note:** To convert these log-probabilities into standard probabilities (0 to 1), use the formula: .

### 3. Using the CLI with --save-details

If you are running LightEval via the command line and want to inspect these probabilities after the run, you **must** use the `--save-details` flag. This exports a Parquet file containing every model generation along with its associated metadata.

```bash
lighteval accelerate \
    --model_args "pretrained=meta-llama/Llama-3-8B" \
    --tasks "path/to/your/task.py" \
    --save-details \
    --output_dir "./results"

```

### 4. Backend-Specific Nuances

* **vLLM Backend:** If you use the vLLM runner, you can pass specific sampling parameters through the `VLLMModelConfig`. Ensure `logprobs` is set to a value greater than 0 (e.g., `logprobs=5` to see the top 5 alternative tokens).
* **Transformers Backend:** LightEval internally uses `model.generate(..., output_scores=True, return_dict_in_generate=True)` when `use_logits` is active, which then populates the `ModelResponse`.

Would you like me to show you how to write a custom metric in LightEval that uses these log-probabilities to calculate model confidence?

Doc
class lighteval.tasks.requests.Doc
<
source
>
( query: strchoices: listgold_index: typing.Union[int, list[int]]instruction: str | None = Noneimages: list['Image'] | None = Nonespecific: dict | None = Noneunconditioned_query: str | None = Noneoriginal_query: str | None = Noneid: str = ''task_name: str = ''fewshot_samples: list = <factory>sampling_methods: list = <factory>fewshot_sorting_class: str | None = Nonegeneration_size: int | None = Nonestop_sequences: list[str] | None = Noneuse_logits: bool = Falsenum_samples: int = 1generation_grammar: None = None )

Parameters

query (str) — The main query, prompt, or question to be sent to the model.
choices (list[str]) — List of possible answer choices for the query. For multiple choice tasks, this contains all options (A, B, C, D, etc.). For generative tasks, this may be empty or contain reference answers.
gold_index (Union[int, list[int]]) — Index or indices of the correct answer(s) in the choices list. For single correct answers,(e.g., 0 for first choice). For multiple correct answers, use a list (e.g., [0, 2] for first and third).
instruction (str | None) — System prompt or task-specific instructions to guide the model. This is typically prepended to the query to set context or behavior.
images (list[“Image”] | None) — List of PIL Image objects for multimodal tasks.
specific (dict | None) — Task-specific information or metadata. Can contain any additional data needed for evaluation.
unconditioned_query (Optional[str]) — Query without task-specific context for PMI normalization. Used to calculate: log P(choice | Query) - log P(choice | Unconditioned Query).
original_query (str | None) — The query before any preprocessing or modification.
# Set by task parameters —
id (str) — Unique identifier for this evaluation instance. Set by the task and not the user.
task_name (str) — Name of the task or benchmark this Doc belongs to.
## Few-shot Learning Parameters —
fewshot_samples (list) — List of Doc objects representing few-shot examples. These examples are prepended to the main query to provide context.
sampling_methods (list[SamplingMethod]) — List of sampling methods to use for this instance. Options: GENERATIVE, LOGPROBS, PERPLEXITY.
fewshot_sorting_class (Optional[str]) — Class label for balanced few-shot example selection. Used to ensure diverse representation in few-shot examples.
## Generation Control Parameters —
generation_size (int | None) — Maximum number of tokens to generate for this instance.
stop_sequences (list[str] | None) — List of strings that should stop generation when encountered. Used for: Controlled generation, preventing unwanted continuations.
use_logits (bool) — Whether to return logits (raw model outputs) in addition to text. Used for: Probability analysis, confidence scoring, detailed evaluation.
num_samples (int) — Number of different samples to generate for this instance. Used for: Diversity analysis, uncertainty estimation, ensemble methods.
generation_grammar (None) — Grammar constraints for generation (currently not implemented). Reserved for: Future structured generation features.
Dataclass representing a single evaluation sample for a benchmark.

This class encapsulates all the information needed to evaluate a model on a single task instance. It contains the input query, expected outputs, metadata, and configuration parameters for different types of evaluation tasks.

Required Fields:

query: The input prompt or question
choices: Available answer choices (for multiple choice tasks)
gold_index: Index(es) of the correct answer(s)
Optional Fields:

instruction: System prompt, task specific. Will be appended to model specific system prompt.
images: Visual inputs for multimodal tasks.
Methods: get_golds(): Returns the correct answer(s) as strings based on gold_index. Handles both single and multiple correct answers.

Usage Examples:

Multiple Choice Question:

Copied
doc = Doc(
    query="What is the capital of France?",
    choices=["London", "Paris", "Berlin", "Madrid"],
    gold_index=1,  # Paris is the correct answer
    instruction="Answer the following geography question:",
)
Generative Task:

Copied
doc = Doc(
    query="Write a short story about a robot.",
    choices=[],  # No predefined choices for generative tasks
    gold_index=0,  # Not used for generative tasks
    generation_size=100,
    stop_sequences=["

End"],
)
Few-shot Learning:

Copied
doc = Doc(
    query="Translate 'Hello world' to Spanish.",
    choices=["Hola mundo", "Bonjour monde", "Ciao mondo"],
    gold_index=0,
    fewshot_samples=[
        Doc(query="Translate 'Good morning' to Spanish.",
            choices=["Buenos días", "Bonjour", "Buongiorno"],
            gold_index=0),
        Doc(query="Translate 'Thank you' to Spanish.",
            choices=["Gracias", "Merci", "Grazie"],
            gold_index=0)
    ],
)
Multimodal Task:

Copied
doc = Doc(
    query="What is shown in this image?",
    choices=["A cat"],
    gold_index=0,
    images=[pil_image],  # PIL Image object
)
get_golds
<
source
>
( )

Return gold targets extracted from the target dict