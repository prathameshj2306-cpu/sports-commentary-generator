

\# Sports Commentary Generator



\[!\[Built with askThili](https://ask.thili.ai/api/v1/public/projects/h9exxdwdvg/badge.svg)](https://ask.thili.ai/b/h9exxdwdvg)



An AI-powered sports commentary generator that creates dynamic commentary for simulated football and cricket events. The project fine-tunes GPT-2 using Low-Rank Adaptation (LoRA) and provides an interactive Streamlit interface for generating event-based commentary.



\## Project Overview



Game developers often need realistic, engaging commentary for events occurring inside sports games. This project explores how a language model can be adapted to generate commentary based on match events and context.



Users can select a sport, specify an event, enter match details, and generate commentary using the trained model.



\## Features



\- Football and cricket commentary generation.

\- Custom dataset containing simulated sports events and commentary.

\- GPT-2 fine-tuning using PEFT LoRA.

\- Interactive Streamlit application.

\- Context-aware prompts using player, team, event and match information.

\- Qualitative evaluation on representative events.

\- LoRA adapter merging into the base model for standalone inference.



\## Dataset



The project uses a custom synthetic dataset containing 200 records:



| Sport | Records |

|---|---:|

| Football | 100 |

| Cricket | 100 |

| \*\*Total\*\* | \*\*200\*\* |



Football event categories include goals, penalties, saves, cards, corners, free kicks, substitutions, missed chances and offsides.



Cricket event categories include sixes, fours, wickets, dot balls, singles, doubles, wides, no-balls, fifties and centuries.



Each record contains:

\- Sport

\- Event type

\- Event description

\- Team

\- Player

\- Match context

\- Commentary



Dataset files are available in the `data/` directory.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Base model:\*\* GPT-2

\- \*\*Model adaptation:\*\* PEFT LoRA

\- \*\*Deep learning:\*\* PyTorch

\- \*\*NLP:\*\* Hugging Face Transformers and Tokenizers

\- \*\*Dataset processing:\*\* Hugging Face Datasets

\- \*\*User interface:\*\* Streamlit

\- \*\*Testing:\*\* Pytest



The implementation uses standard PEFT LoRA adapter injection, training and merging.



\## Project Structure



```text

sports-commentary-generator/

├── app/

│   ├── core.py

│   ├── llm.py

│   ├── main.py

│   └── storage.py

├── data/

│   ├── football.jsonl

│   ├── cricket.jsonl

│   └── combined\_dataset.jsonl

├── models/

│   └── sports-commentary-lora/

├── tests/

│   ├── test\_core.py

│   └── test\_smoke.py

├── .github/

│   └── workflows/

│       └── ci.yml

├── requirements.txt

├── Dockerfile

└── README.md

```



The merged model is generated locally and is excluded from Git because of its large file size.



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/prathameshj2306-cpu/sports-commentary-generator.git

cd sports-commentary-generator

```



\### 2. Create a virtual environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



\*\*Linux/macOS:\*\*



```bash

python3.12 -m venv .venv

source .venv/bin/activate

```



\### 3. Install dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Python 3.12 is recommended.



\## Running the Application



The trained LoRA adapter is included in the repository.



Start the Streamlit application:



```bash

streamlit run app/main.py

```



Open the local URL displayed in the terminal, usually:



```text

http://localhost:8501

```



Select a sport, choose an event type, enter the event description and match details, and click the generate button.



The first model load may take some time because GPT-2 needs to be downloaded if it is not already cached.



\## Training the LoRA Adapter



To train the adapter again using the complete dataset:



```bash

python -c "from app.core import milestone\_2; print(milestone\_2())"

```



The training configuration uses GPT-2, five epochs and a learning rate of 0.0001.



The trained adapter is saved in:



```text

models/sports-commentary-lora/

```



\## Evaluation



Run the qualitative evaluation:



```bash

python -c "from app.core import milestone\_4; results = milestone\_4(); print('Total evaluated:', len(results))"

```



The evaluation checks representative football and cricket events, including goals, penalties, sixes and wickets.



The generated commentary is compared with the corresponding reference commentary.



\*\*Evaluation limitation:\*\* These representative examples are selected from the same dataset used during training. This is a qualitative review, not an independent held-out evaluation or an objective accuracy measurement.



\## LoRA Adapter Merging



Merge the trained adapter into the GPT-2 base model:



```bash

python -c "from app.core import milestone\_5; print(milestone\_5())"

```



The merged model is saved locally in:



```text

models/sports-commentary-merged/

```



This allows inference using the merged model without separately loading the LoRA adapter.



The merged model is excluded from Git because it is approximately 475 MB.



\## Running Tests



Execute the project test suite:



```bash

pytest -v

```



The tests cover dataset validation, preprocessing, training, application setup, evaluation and adapter merging.



\## Limitations



\- The dataset contains 200 synthetic examples and has limited diversity.

\- Generated commentary may sometimes be repetitive or inaccurate.

\- The model's performance depends on the quality and coverage of training examples.

\- Evaluation is qualitative and uses examples from the training dataset.

\- CPU-based training and inference can take longer than GPU execution.



\## Future Improvements



\- Expand the dataset with more diverse sports events and commentary styles.

\- Introduce independent training, validation and test splits.

\- Add quantitative evaluation metrics and human feedback.

\- Improve generation quality through prompt engineering and data augmentation.

\- Explore larger language models and GPU-based training.

\- Add more sports and event categories.



\## Acknowledgements



Built as an askThili project.



\[Built with askThili](https://ask.thili.ai/b/h9exxdwdvg)



