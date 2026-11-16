# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/h9exxdwdvg/badge.svg)](https://ask.thili.ai/b/h9exxdwdvg)

This project provides game developers with a powerful tool to generate dynamic and context-aware sports commentary. By adapting a pre-trained language model with LoRA, it creates realistic commentary snippets for simulated sports events like goal alerts or score changes, enhancing the immersive experience of games.

## What you are building

**Goal.** You will build a tool for game developers that uses a LoRA-adapted language model to generate dynamic and context-aware sports commentary snippets for simulated events.

**Done when.** Your project is complete when you have a trained LoRA adapter on a pre-trained LLM, a Streamlit interface that takes event descriptions and generates commentary, and the adapter is merged for efficient inference, proving the LoRA fine-tuning technique.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of simulated sports events and commentary pairs.**  
      Hint: Use text preprocessing and data loading techniques to prepare your dataset.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the lfllm-lflora technique for LoRA adapter injection and training.
- [ ] **3. Build a Streamlit interface to input event descriptions and generate commentary.**  
      Hint: Utilize Streamlit components to build an interactive user interface for input and output.
- [ ] **4. Test the model's ability to produce relevant and engaging commentary.**  
      Hint: Perform qualitative and quantitative analysis to assess commentary quality.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Implement the lfllm-lflora adapter merging functionality for efficient deployment.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Rigorously test and evaluate the model's ability to produce relevant and engaging commentary across a wide range of simulated events.
- Explore advanced prompt engineering techniques or dataset augmentation strategies to further enhance the dynamism and context-awareness of the generated commentary.
