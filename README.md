# Visual Search — research notebooks

Research I did in **December 2023 and January 2024** working out how to build visual search for
ecommerce: upload a product photo, get visually similar products back.

This is the working-out, not a finished product. It later informed
**[Guidance for Visual Search on AWS](https://github.com/aws-solutions-library-samples/guidance-for-visual-search-on-aws)**
in the AWS Solutions Library, which is the cleaned-up, deployable version of the same idea.

## The question it was built to answer

In December 2023, **Amazon Titan Multimodal Embeddings had just launched** and vision-capable LLMs
hadn't arrived yet. So "what should embed the product images?" was genuinely open: CLIP, which the
research community had standardised on, or the new managed service.

The repository builds the same pipeline both ways so the two can be compared side by side:

| Notebook | Embeddings |
|---|---|
| `vsJointFlip2.ipynb` | **CLIP** (`openai/clip-*`) — joint image and text embedding space |
| `vsJointFlip2Bedrock.ipynb` | **Amazon Titan** (`amazon.titan-embed-image-v1`, `amazon.titan-embed-text-v1`) |
| `vs-notebook.ipynb` | the cleaner walkthrough written up for the blog draft |
| `AttsfromEmbs.ipynb` | deriving product attributes from embeddings |
| `gen_desc.ipynb` | product description generation and loading the catalogue into DynamoDB |

Both paths index into FAISS and expose search through a small service in
`visual-search-service/` (Streamlit front end, Dockerfile, `services.py`).

**An honest gap:** the notebooks construct HNSW and IVF indices and set parameters like
`ef_search`, but **no recall or latency measurements were recorded**. The comparison was run and
the numbers weren't kept. Treat this as two documented implementations, not a benchmark — if you
want the numbers, the indices are reproducible from the notebooks.

## Running it

You'll need to supply two things the repository deliberately doesn't ship:

**The dataset.** Download [Flipkart Products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products)
from Kaggle as `flipkart_com-ecommerce_sample.csv` in the repository root. It isn't included here —
redistributing someone else's dataset is their call, not mine.

**The indices.** The `.index` files are build artifacts and are not versioned. Run the notebooks to
regenerate them.

Everything else — `code/inference.py` for the SageMaker endpoint, `scripts/helper_functions.py`,
the architecture diagrams in `ImageSearch-Phase1*.png` — is here.

## `blog.md`

A draft blog post, *Enabling Visual Search for Ecommerce with CLIP and Amazon SageMaker*. It's
unfinished: several sections are still outline bullets rather than prose. Kept because the
structure and the architecture walkthrough are useful, and because it's the bridge between these
notebooks and the published guidance.

## What this is, and what it isn't

Research notebooks from a point in time, kept because the question they answer was real then and
the answer is now obvious. Specifically:

- model identifiers from late 2023 — several of the Titan ones have since been superseded
- no evaluation harness, no recorded measurements
- two versions of the search service (`visual-search-service/` and `src/visual-search-service/`)
  that drifted apart during the work and were never reconciled
- account identifiers and profile names replaced with placeholders

For current work on the same ground, the deployable version is the AWS Solutions Library guidance
linked above.

---

A personal project. Not an official AWS project, and not supported by AWS.
