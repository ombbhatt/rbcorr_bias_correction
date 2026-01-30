# RBCorr: Response Bias Correction in Language Models

This repository serves as the codebase for our work on exploring LogProbs-based bias correction in LLMs. You can find the paper here: 

**Abstract:**

Language models (LMs) are known to be prone to response biases, which present as option preference biases in fixed-response questions. It is therefore imperative to develop low-cost and effective response bias correction methods to improve LM performance and enable more accurate evaluations of model abilities. Here, we propose a simple response bias correction strategy, RBCorr, and test it on 12 open-weight language models using yes-no, entailment, and multiple choice questions. We show that response bias is prevalent in LMs pre-correction and that RBcorr effectively eliminates bias and boosts model performance. We also explore the generalizability of bias behavior across models, datasets, and prompt formats, showing that LogProbs-based correction is highly dependent on all three of these aspects. Overall, RBcorr is an easy-to-use method that can boost the performance of smaller LMs and ensure that LM performance on closed-response benchmarks aligns more closely with their true capabilities.

**NOTE-1:** The internal name of the RBCorr method used across the codebase is `"specific"`. Please keep in mind that any filename or argument called "specific" henceforth refers to the RBCorr method.

---

## Setup

1. Clone the repository
2. Create a conda environment: `conda create -n rbcorrenv python=3.11` and then `conda activate rbcorrenv`
3. Install required dependencies into conda environment: `pip install -r requirements.txt` and then `pip install -U transformers`

That's it! You are ready to run inference, perform corrections and run analysis scripts in this codebase.

---

## Code Organization

The repository contains four home folders:
1. **`data/`:** Contains all datasets of questions used to compute LLM LogProbs and run bias correction methods. 
2. **`results/`:** Contains JSON-formatted data on the accuracy, bias, and related performance statistics for all chosen correction methods across our entire test suite of models, datasets, and prompt formats.
   1. These are split into subfolders according to the correction method and question type. We have `corrmethods = [batchcalib, contextcalib, specific]` and `qtypes = [yesno, nli, mcq]`, and thus the subfolders containing the JSON files are named as `results/{corrmethod}_{qtype}_per_TVD/`.
   2. This folder also contains outputs of plotting and analysis scripts (in `results/plot_outputs/` and `results/table_outputs/`).
3. **`src/`:** Contains main implementation of logic to load model --> feed datasets --> extract and store plain inference LogProbs values in CSV files in an `outputs/` folder --> apply a chosen correction method using those plain inference values --> store the corrected LogProbs into another CSV file in `outputs/`.
   1. `src/unified_transprompt_transmodel_transdata.py` serves as the primary script where all options for running some configuration of model, dataset, prompt, and correction method can be specified via CLI flags.
   2. Given the large number of potential combinations of models, datasets, prompts, and correction methods, we provide shell scripts for running them easily (see `src/run_{method}_shellscripts/`). These automatically run the primary script in a loop across all valid combinations of setup configurations, for the correction method and question type of your choice! **However, please see NOTE-2.**
4. **`scripts/`:** Contains three sub-folders.
   1. `scripts/correction_metrics_processors/` has the scripts which generate the JSON-formatted performance statistics which get saved in the aforementioned `results/` folder (given that the `outputs/` home folder exists and contains the required LogProbs CSV file to read and process).
   2. `scripts/transfer_analyses/` has the scripts to analyse the efficacy of RBCorr correction in 'transfer correction' cases.
   3. `scripts/misc_analyses_scripts/` has scripts for all other plots and analyses presented in the paper.

**NOTE-2:** Unless you want to re-create the full per-question LogProbs CSV files for our test suite, or run new models or datasets (either of which require compute power and storage), you do NOT need to run this primary script in `src/` at all! The only purpose of these LogProbs CSV files is to serve as the input to the metrics processor scripts (in `scripts/`), in order to ultimately generate the JSON-formatted performance statistics. We already provide all of the JSON files that we generated across our test suite in `results/`!

---

## Datasets

We created custom datasets in the `data/` folder derived from subsets of various existing datasets (more details for each dataset provided in the paper!). These focus on modifying the orginal datasets to enforce class-balance, standard formatting across datasets, and standardized single-token response format based on question-type.

* Yes-No Datasets (2-Choice; `Yes/No`):
  1. ARITH -- `data/arithynq-scripts-data/arith-ynq-big.csv`
  2. BABI -- `data/babiynq-scripts-data/babi-ynq-big.csv`
  3. COMPS -- `data/compsynq-scripts-data/comps_yn_rand_2prop_2100.csv`
  4. EWOK -- `data/ewokynq-scripts-data/t2q_nodup_nominpairs/processed_t2q_all_domains.csv`
* NLI Datasets (3-choice; `0/1/2`):
  1. SNLI -- `data/snli-scripts-data/snli_1.0/snli_1.0_test_balanced_smaller.csv`
  2. MNLI -- `data/mnli-scripts-data/multinli_1.0/multinli_1.0_dev_matched_balanced_genre.csv`
* MMLU Datasets (4-choice; `A/B/C/D`):
  1. HUMANITIES -- `data/mmlu-scripts-data/HUMANITIES/*`
  2. OTHERS -- `data/mmlu-scripts-data/OTHERS/*`
  3. SOCIAL SCIENCES -- `data/mmlu-scripts-data/SOC_SCI/*`
  4. STEM -- `data/mmlu-scripts-data/STEM/*`
 
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

## Reading Results

### JSON Nesting Structure

The JSON files in `results/` follow a standard nested structure to organize the performance statistics across multiple setup configurations. We will take one example file from the RBCorr results, `results/specific_yesno_per_median_TVD/fewshot_Falcon_ARITH.json`, and describe the nesting structure. This can then be generalized to other RBCorr files; the BC and CC results also follow a similar but simpler structure given the lack of transfer correction cases.

`results/specific_yesno_per_median_TVD/fewshot_Falcon_ARITH.json` starts like this:
```
{
  "fewshot": {
    "PER_YESNO": {
      "ARITH-fromARITH": {
        "Falcon": {
          "Falcon3-3B-Base": {
            "20": { ... }
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
* The final sixth-level key denotes the calibration set size that was used when running the RBCorr correction. For all same-condition (i.e. non-transfer) RBCorr configurations, we run the correction and store results using a fixed set of multiple calibration set sizes: `[20, 50, 100, 500, 1000]`. For each calibration set size, we run RBCorr correction 100 times using randomly-sampled calibration sets of that size and report the aggregate results, for a robust idea of the performance effects of applying the correction. The transfer correction cases do the same, but use only one calibration set size of `500`.

### Reported Metrics
The sixth-level key entails a block of values that looks like this:
```
  "best_tvd": 0.0016666666666666774,
  "best_tvd_model_dist": "{'Yes': 0.5016666666666667, 'No': 0.49833333333333335}",
  "worst_tvd": 0.17,
  "worst_tvd_model_dist": "{'Yes': 0.67, 'No': 0.33}",
  "mean_tvd": 0.03992289151007473,
  "mean_tvd_model_dist": "{'Yes': 0.46, 'No': 0.54}",
  "median_tvd": 0.03166666666666665,
  "median_tvd_model_dist": "{'Yes': 0.5316666666666666, 'No': 0.4683333333333333}",
  "std_tvd": 0.03380431252538588,
  "q25_tvd": 0.012916666666666674,
  "q25_tvd_model_dist": "{'Yes': 0.48833333333333334, 'No': 0.5116666666666667}",
  "q75_tvd": 0.05624999999999998,
  "q75_tvd_model_dist": "{'Yes': 0.5566666666666666, 'No': 0.44333333333333336}",
  "best_run_acc": 0.875,
  "worst_run_acc": 0.8058333333333333,
  "mean_acc": 0.8510585636546931,
  "median_acc": 0.8516666666666667,
  "std_acc": 0.013900296971728789,
  "q25_acc": 0.8459935897435897,
  "q75_acc": 0.8595833333333334,
  "raw_tvd": 0.06916666666666668,
  "raw_acc": 0.8758333333333334,
  "raw_model_dist": "{'Yes': 0.5691666666666667, 'No': 0.43083333333333335}",
  "ground_truth_dist": "{'Yes': 0.5, 'No': 0.5}",
  "num_calib_sets": 99
```
Since we run 100 RBCorr correction for each calibration set size using randomly-sampled calibration sets, we are able to report both single-run and aggregate results on accuracy and bias. The values containing `acc` denote accuracy and the ones containing  `tvd` denote bias value, both ranging from 0-1. For TVD, 0 indicates identical distribution to the ground-truth (which is always uniform distribution), i.e., complete elimination of bias. Across the 100 runs, we report the single best and worst accuracy and tvd values, as well as the mean, median, 25th and 75th percentile values across all runs. We also report the response distribution of the model for the available answer choices before applying any correction (`raw_model_dist`) and after applying RBCorr (other `model_dist` values above). 

In general, we want to see a higher median accuracy and lower median tvd compared to the `raw` counterparts to confirm that RBCorr was able to reliably increase accuracy and decrease bias.

---

## Takeaway Findings

<img width="4615" height="3917" alt="scatterplot_visualization (21)" src="https://github.com/user-attachments/assets/9b5b29a1-401d-45c8-ad29-34707797687f" />

Scatterplots showing per-model bias (TVD; $\downarrow$ is better) and accuracy (\%; $\uparrow$ is better) before ($\bullet$) vs. after ($\times$) applying RBCorr correction. We show results on one dataset per each question-type; the bottom-right plot shows results averaged across all ten datasets.

<img width="986" height="730" alt="bc_cc_rbcorr_table" src="https://github.com/user-attachments/assets/87331729-b162-4e12-8cec-fc416053d9f8" />

Comparison between Contextual Calibration (CC), Batch Calibration (BC) and our method (RBCorr), on all datasets using the small Llama3.1 model pair and fewshot prompt format. Best accuracy change (highest $\uparrow$ / lowest $\downarrow$) is shown in bold and best TVD change (highest $\downarrow$ / lowest $\uparrow$) is italicized in each row.
