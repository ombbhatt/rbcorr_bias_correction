# RBCorr: Response Bias Correction in Language Models

This repository serves as the codebase for our work on exploring LogProbs-based bias correction in LLMs. You can find the paper here: 

---

## Setup

1. Clone the repository
2. Create a conda environment: `conda create -n rbcorrenv python=3.11` and then `conda activate rbcorrenv`
3. Install required dependencies into conda environment: `pip install -r requirements.txt` and then `pip install -U transformers`

That's it! You are ready to run inference, perform corrections and run analysis scripts in this codebase.

---

## Datasets

We created custom datasets in the `data/big_data/` folder derived from subsets of various existing datasets (more details for each dataset provided in the paper!). These focus on modifying the orginal datasets to enforce class-balance, standard formatting across datasets, and standardized single-token response format based on question-type. Feel free to use them if you are looking for simple easy-to-parse questions for your testing purposes!

Each dataset file consists of 1200 total questions, and we maintain as much equality in samplign from subdomains of the original dataset as possible, so that each of these datasets are smalleryet faithful representations of the original datasets they were derived from.

* Yes-No Datasets (2-Choice; `Yes/No`):
  1. ARITH -- `arith-ynq-big.csv`
  2. BABI -- `babi-ynq-big.csv`
  3. COMPS -- `comps-ynq-big.csv`
  4. EWOK -- `ewok-ynq-big.csv`
* NLI Datasets (3-choice; `0/1/2`):
  1. SNLI -- `snli-nli-balanced.csv`
  2. MNLI -- `mnli-nli-balanced.csv`
* MMLU Datasets (4-choice; `A/B/C/D`):
  1. HUMANITIES -- `mmlu-scripts-data/HUMANITIES/HUMANITIES_sampled.csv`
  2. OTHERS -- `mmlu-scripts-data/OTHERS/OTHERS_sampled.csv`
  3. SOCIAL SCIENCES -- `mmlu-scripts-data/SOCIAL_SCI/SOCIAL_SCI_sampled.csv`
  4. STEM -- `mmlu-scripts-data/STEM/STEM_sampled.csv`
 
---

## Models, Prompt Formats, Correction Methods

We use the following 12 models among 3 model families, sourced from huggingface:

```
falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
gemma3_models = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
```

We use the following three prompt formats:

1. Zeroshot: where only the test question is presented,
2. Instruction-only: where a one-line task instruction precedes the test question,
3. Fewshot: which provides a one-line task instruction and two example question-answers before presenting the test question.

We test three correction methods in total, described in detail in the paper: 

1. RBCorr, our novel correction method,
2. Batch Calibration (BC) [(Zhou et al., 2024)](https://arxiv.org/abs/2309.17249)
3. Contextual Calibration (CC) [(Zhao et al., 2021)](https://arxiv.org/abs/2102.09690)

---

## Code Organization

The repository contains four home folders:
1. **`data/`:** Contains all datasets of questions used to compute LLM LogProbs and run bias correction methods. 
    1. `data/big_data/` contains the datasets used for the paper, i.e. 1200 test questions per dataset.
    2. `data/tiny_data/` contains mini-versions (200 questions per dataset) for testing purposes.
    3. `data/precompute_cc_logprobs/` contains the code and csv files that contain the per-option token loghprobs for all test models for all datasets which are used to perform the Contextual Calibration correction method. The correction method is summarized in the paper.

2. **`src/`:** Contains main implementation of logic to load model --> feed datasets --> extract and store plain inference LogProbs values in CSV files in an `outputs/` folder --> apply a chosen correction method using those plain inference values --> store the corrected PER-ITEM LogProbs into another CSV file in `outputs/`. All data discussed in the paper is located within the "Mar-23-2026" `DATE` subfolders, wherever applicable.
   1. `src/plain_inference.py` runs the base inference required to get the per-item answer option logprobs. Accepts a model name, dataset name, and prompt type as input parameters. Run with `--help` to see all input choices for models, datasets, and prompt types. Running this will create a new `outputs/` root-level folder, which then contains CSV outputs nested under `{DATE}/{prompt-type}/{dataset}/{question-type}plain/{model-family}/{domain}/{model-name}_results.csv`. However, please see **NOTE-1** below.
   2. `src/cc_impl.py` and `src/bc_impl.py` use the plain per-item loogprobs to perform the Contextual Calibration and Batch Calibration correction methods respectively. These do not accept input arguments; the script expects that all plain logprobs exist for all models, datasets, and prompt types. If the plain logprobs for any configuration is missing, please comment out the specific model, dataset, or prompt type from the script to successfully generate the corrected per-item logprobs. These results are also saved the `outputs/` folder, nested in the same way as the plain inference results, except that the subfolder under `.../{dataset}/...` will now be `{question-type}cccorr` or `{question-type}bccorr_n{N}_k{K}`, where `N` and `K` are values for the batch size and k-fold count set for the BC method.
   3. `rbcorr_impl.py` and `rbcorr_transfer_impl.py` and `rbcorr_transfer_loop.sh` contain the code required to run our correction. As with the other correction implementation scripts, these expect the plain inference logprobs for all model-dataset-prompt configurations to exist, and missing ones must be manually commented out in the lists defined in the code. Given the hundreds of RBCorr "transfer correction" configurations possible, the bash file is designed to run the transfer implementation script in a loop across all of the valid transfer configurations across all three modalities. Again, the results from these are saved in the `outputs/` folder, and the the subfolder under `.../{dataset}/...` will now be `{question-type}rbcorr_n{N}_k{K}`, where `N` and `K` are values for the batch size and k-fold count set for the `RBCorr` method.

3. **`extract_metrics/`:** This folder contains the code to take the per-item logprobs from the `outputs/` folder and generate structured dataset-level model performance data in the `results/` folder. 
    1. `config.py` contains the model, dataset, and prompt type list, as well as the specification of the batch size and k-fold count, each of which must be set manually according to the configuration used to generate the `BC` and `RBCorr` per-item logprobs data. 
    2. Then, simply run `python main.py [--method cc|bc|rb|all] [--dry-run]` to either preview or run the metrics calculation and generate the JSON files in the `results/` folder. Again, running these scripts requires the item-level logprobs to exist, and are not required to run to reproduce our results, as explained in **NOTE-1** below. The resulting JSON files are provided in the `results/` folder.

4. **`results/`:** Contains JSON-formatted data on the accuracy, bias, and related performance statistics for all chosen correction methods across our entire test suite of models, datasets, and prompt formats. The results discussed in the paper are located in `results/Mar-23-2026`.
   1. The JSON files are split into subfolders according to the correction method and question type. We have `corrmethods = [bc, cc, rb]` where `rb` is our method, and `qtypes = [yn, nli, mcq]`, and thus the subfolders containing the JSON files are named as `results/{corrmethod}_{qtype}/`. The JSON files within them are named using the convention `{prompt-type}_{model-family}_{dataset}.json`.
   2. The `comparison...` and `ttest...` CSV files are generated based on the result JSONs from the `results/` folder, using analysis scripts from the  `plots_tables_scripts/` folder; the plots are subsequently created using these CSV files also using scripts from the aforementioned folder.

5. **`plots_tables_scripts/`:** Contains the scripts that use the `results/` JSON files to create the plots and tables used to compare and analyze model performance and correction efficacy. Not explained here for brevity.

**NOTE-1:** Unless you want to re-create the full per-question LogProbs CSV files for our test suite, or run new models or datasets (either of which require compute power and storage), you do NOT need to run this primary script in `src/` at all! In fact, we do not even include the `outputs/` folder containing the per-item plain logprobs, since it takes up large amounts of data. The only purpose of these LogProbs CSV files is to serve as the input to the metrics processor scripts (in `extract_metrics/`), in order to ultimately generate the JSON-formatted performance statistics in `results/`. We already provide all of the JSON files that we generated across our test suite in `results/`! 

---

## Reading Results

### JSON Nesting Structure

The JSON files in `results/` follow a standard nested structure to organize the performance statistics across multiple setup configurations. We will take one example file from the RBCorr results, `results/rb_yn/fewshot_Falcon_ARITH.json`, and describe the nesting structure. This can then be generalized to other RBCorr files; the BC and CC results also follow a similar but simpler structure given the lack of transfer correction cases.

`results/rb_yn/fewshot_Falcon_ARITH.json` starts like this:
```
{
  "fewshot": {
    "YESNO": {
      "ARITH-fromARITH": {
        "Falcon": {
          "Falcon3-3B-Base": {
            "60": { ... }
            ...
```
* Starting from the outermost level, the first and second-level keys are self-explanatory -- they describe the prompt format and the question type of the configuration that we are currently seeing the results for. 
* The third-level key defines the dataset and all information about transfer correction in cases where transfer correction has been processed.
    1. Notice how it says `"ARITH-fromARITH"`; this is to indicate potential cross-dataset RBCorr correction. Since we are looking at `fewshot_Falcon_ARITH.json`, we will only see cross-dataset keys of the format `"ARITH-from{other_yesno_dataset}"`.
    2. This key will also flag whether the nested result is for a cross-model correction. When it is a same-model correction, as above, the third-level key will not contain any model name. When it is a cross-model correction, the key will be suffixed with the target and source model names, e.g., `"ARITH-fromARITH_Falcon3-10B-Instruct_fromFalcon3-10B-Base"`. This means that the target configuration used `Falcon3-10B-Instruct` to generate LogProbs values. Since we are looking at `fewshot_Falcon_ARITH.json`, we will only see cross-model keys of the format `"ARITH-fromARITH_{FalconModelA}_from{FalconModelB}"`.
    3. This key will also flag whether the nested result is for a cross-prompt correction. When it is a same-prompt correction, as above, the third-level key will not contain any prompt. When it is a cross-prompt correction, the key will be suffixed with the target and source prompt formats, e.g., `"ARITH-fromARITH_fewshot_frominstronly"`. This means that the target configuration used `fewshot` formatted inputs to generate LogProbs values.
    4. We will never see a third-level key with more than one kind of transfer occurring. For example, we will never see both cross-dataset and cross-model enabled, which would theoretically yield a key that looks like `"ARITH-fromBABI_Falcon3-10B-Instruct_fromFalcon3-10B-Base`"
* The fourth-level key denotes the model family. This could be `Falcon`, `Gemma3`, or `Llama3`.
* The fifth-level key denotes the specific model within the model family. Here, we see `Falcon3-3B-Base`. Note that in the case of cross-model correction, this denotes the target configuration model. If we see a third-level key that looks like `"ARITH-fromARITH_{FalconModelA}_from{FalconModelB}"`, then the fifth-level key will strictly be `"{FalconModelA}"`.
* The final sixth-level key denotes the calibration set size that was used when running the RBCorr correction. For all same-condition (i.e. non-transfer) RBCorr configurations, we run the correction and store results using a calibration set size of `60`. We run RBCorr correction `2` times using randomly-sampled calibration sets of that size and report the aggregate results, for a robust idea of the performance effects of applying the correction. The transfer correction cases do the same, but using a calibration set size of `180`.

### Reported Metrics
The sixth-level key entails a block of values that looks like this:
```
  "acc": 0.8683333333333333,
  "tvd": 0.04500000000000001,
  "rsd": 0.05182341650671784,
  "model_dist": "{'Yes': 0.545, 'No': 0.455}",
  "mean_acc": 0.8710526315789474,
  "median_acc": 0.8710526315789474,
  "std_acc": 0.0026315789473684292,
  "best_run_acc": 0.8736842105263158,
  "worst_run_acc": 0.868421052631579,
  "mean_tvd": 0.0324561403508772,
  "median_tvd": 0.0324561403508772,
  "std_tvd": 0.011403508771929846,
  "mean_rsd": 0.03722161372763781,
  "median_rsd": 0.03722161372763781,
  "std_rsd": 0.012979189485213559,
  "raw_acc": 0.8691666666666666,
  "raw_tvd": 0.08416666666666664,
  "raw_rsd": 0.09683604985618409,
  "raw_model_dist": "{'Yes': 0.5841666666666666, 'No': 0.41583333333333333}",
  "ground_truth_dist": "{'Yes': 0.5, 'No': 0.5}",
  "num_calib_sets": 2
```
Since we run 2 RBCorr correction for each calibration set size using randomly-sampled calibration sets, we are able to report both single-run and aggregate results on accuracy and bias. The values containing `acc` denote accuracy and the ones containing  `tvd` denote bias value, both ranging from 0-1. For TVD, 0 indicates identical distribution to the ground-truth (which is always uniform distribution), i.e., complete elimination of bias. Across the 100 runs, we report the single best and worst accuracy and tvd values, as well as the mean, median, 25th and 75th percentile values across all runs. We also report the response distribution of the model for the available answer choices before applying any correction (`raw_model_dist`) and after applying RBCorr (other `model_dist` values above). 

In general, we want to see a higher or equal median accuracy and lower median tvd compared to the `raw` counterparts to confirm that RBCorr was able to reliably decrease bias while preserving or increasing accuracy.

---

## Takeaway Findings



Scatterplots showing per-model bias (TVD; $\downarrow$ is better) and accuracy (\%; $\uparrow$ is better) before ($\bullet$) vs. after ($\times$) applying RBCorr correction. We show results on one dataset per each question-type; the bottom-right plot shows results averaged across all ten datasets.
