## AutoPAC - Automated Plan and Code Synthesis for Continual Machine Learning Ideation


## Note ##
**The experiments were conducted using LLMs; hence, they may not be precisely reproducible, as LLMs are stochastic and produce varying outputs even with the same inputs**.

Before proceeding, please ensure to insert your API keys into the appropriate fields in the JSON file. Specifically, update the values for `GeminiAPIKEY` and `GPTAPIKEY` with your respective keys.

After downloading the JSON file, locate the placeholders for API keys and replace them with your actual keys. This step is crucial for the proper functioning of the project, especially if you plan to interact with external services or APIs.

Once you have updated the API keys, you can proceed with running the experiment or utilizing the functionalities provided by the project.


#### Stage 1: High-Level Idea Generation
- **Inputs**: 
  - Paper: A collection of papers stored in `/Docs/outputs`.
  - Model: Any pre-trained language model (LLM) used for idea generation.
  - Dataset: Relevant datasets for training and evaluation.
- **Outputs**:
  - Idea extracted from Paper: Ideas distilled from the papers stored in `/Docs/Outputs/Initial idea`.
  - Idea generated using the Paper + Model + Dataset: Enhanced ideas generated through combining paper content, language model, and datasets stored in `/Docs/outputs/Refined`.
  -Codes,papers and prompts used for High-Level Idea generation is present in `\High-level Idea Generation` folder.

#### Stage 2: Base Code Generation
- **Inputs**:
  - Idea generated of the Paper of Intereset using `/project_3` code.
- **Outputs**:
  - Baseline code: Initial code generated based on the idea, stored in the main directory in inside the folder `{experiment_name}{count}` as `Base_code.py`.

#### Stage 3: Iterative Code Generation
- **Inputs**:
  - Base code generated in Stage 1.Refined Idea generated using the paper of interest,model and the dataset.
- **Outputs**:
  - Iterative code: Enhanced code incorporating the novel ML technique or idea, stored as `final_code.py` in `{experiment}{number}/`.


### Overview:

This project aims to automate the generation of code snippets from research papers. By utilizing advanced machine learning models such as Gemini and GPT, along with specific configurations outlined in a JSON file, the system generates code relevant to the research paper's content.

### How to Use:

1. **Update JSON Configuration:**
   - Modify the provided JSON file with the necessary parameters.
   - Adjust parameters such as `ExpName`, `FolderName`, `ErrorCorrectionReRuns`, `GeminiModel`, `GPTModel`, `ClaudeModel`, `InstructionGenerationModel`, `CodeGenerationModel`, `SELF_CONSISTENCY`, `RunCount`, `ideaFilePath`, `contextFilePath`, `StrategyPath`, `librariesPath`, `sampleModelCode`, and `datasetLoadingCode` according to your requirements.

2. **Execute the Script:**
   - Run the script, ensuring that the path to the JSON file is correctly specified.

### JSON File:

- **ExpName:** Title of the project.
- **FolderName:** Main folder where experiment data is stored.
- **ErrorCorrectionReRuns:** Maximum number of attempts given to the debugger to make the code error-free.
- **GeminiModel:** Name of the Gemini model to be used.
- **GPTModel:** Name of the GPT model to be used.
- **ClaudeModel:** Name of the Claude model to be used. (Note: Claude functionality is currently unavailable in India.)
- **API:** Insert your own api keys
- **InstructionGenerationModel:** Choose between "GPT" and "Gemini".
- **CodeGenerationModel:** Specify "GPT" for now.
- **SELF_CONSISTENCY:** Higher values provide better results but are more computationally expensive.
- **RunCount:** Set to 1 for each experiment. Increment for multiple re-runs.
- **ideaFilePath:** Location of the "idea" text file.
- **contextFilePath:** Location of the "Context" text file.
- **StrategyPath:** Location of the "Strategy" text file.
- **librariesPath:** Location of the recommended libraries text file.
- **sampleModelCode:** Sample code of the model to be used.
- **datasetLoadingCode:** Sample dataset loading code to be fed into the new model.
- **packages:** Relevant packages and libraries.
- **model_refined_methodology:** High level idea which includes the paper concerned, the model, and the dataset.
- **model_methodology:** Refined mode **model_refined_methodology** 
- Also add paths to where you want to store the codes.


### Additional Notes:

- Ensure that the specified paths in the JSON file point to the correct locations of respective files.
- Experiment with different models and configurations to optimize code generation results.

For any inquiries or issues, please contact the project maintainers.
