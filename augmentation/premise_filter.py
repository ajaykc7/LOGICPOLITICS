import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from logical_fallacy import MNLIDataset, get_unique_labels, replace_masked_tokens

FALLACY_HASH = {
    "p_emotion": "apeal to emotion",
    "p_dilemma": "false dilemma",
    "p_relevancy": "fallacy of relevance",
    "p_intention": "intentional"
}

class CustomMNLIDataset(MNLIDataset):

    def __init__(self, tokenizer_path, train_df, val_df, label_col_name, map='base', test_df=None, fallacy=False,
                 undersample_train=False, undersample_val=False, undersample_test=False, undersample_rate=0.04,
                 train_strat=1, test_strat=1, multilabel=False):
        # strat is used in convert_to_mnli function
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        torch.manual_seed(0)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        special_tokens_dict = {
            'additional_special_tokens': ["[A]", "[B]", "[C]", "[D]", "[E]", "[F]", "[G]", "[H]", "[I]"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.train_data = None
        self.val_data = None
        self.label_col_name = label_col_name
        self.map = map
        self.mappings = pd.read_csv("logical_fallacy/data/mappings.csv")
        self.multilabel = multilabel
        if fallacy:
            self.unique_labels, self.counts_dict = get_unique_labels(pd.concat([train_df, val_df, test_df]),
                                                                     self.label_col_name,
                                                                     multilabel)
            self.total_count = 0
            for count in self.counts_dict.values():
                self.total_count += count
        self.init_data(fallacy, undersample_train, undersample_val, undersample_test,
                       undersample_ratio=undersample_rate, train_strat=train_strat, test_strat=test_strat)

    def convert_to_mnli(self, df, undersample=False, undersampling_rate=0.02, strat=1):
        """
        strat=1 -> only original article
        strat=2 -> only masked article
        strat=3 -> both
        """
        data = []
        for i, row in df.iterrows():
            for label in self.unique_labels:
                entry = [row['source_article']]
                if self.map == 'base':
                    entry.append("This is an example of %s logical fallacy" % label)
                elif self.map == 'masked-logical-form':
                    # print(self.mappings[self.mappings['Original Name'] == label]['Masked Logical Form'])
                    form = replace_masked_tokens(list(self.mappings[self.mappings['Original Name'] == label]
                                                      ['Masked Logical Form'])[0])
                    entry.append("This article matches the following logical form: %s" % form)
                if (self.multilabel is False and label == row[self.label_col_name]) or (
                        self.multilabel is True and label in row[self.label_col_name]):
                    entry.append("entailment")
                else:
                    entry.append("contradiction")
                weight = (self.total_count / self.counts_dict[label]) / 10
                if entry[-1] == "entailment":
                    weight *= 12
                entry.append(weight)
                entry.append(label)
                if strat % 2:
                    data.append(entry)
                if strat > 1:
                    entry1 = [replace_masked_tokens(row['masked_articles']), entry[1], entry[2], entry[3], entry[4]]
                    if entry1[0] != entry[0] or strat == 3:
                        data.append(entry1)

        return pd.DataFrame(data, columns=['sentence1', 'sentence2', 'gold_label', 'weight','logical_fallacy'])

def filter(model, test_loader, device, tokenizer):
    """
    Get predictions of premises with each of the 4 fallacies

    Parameters
    ----------
    model:
        the pretrained model
    test_loader:
        DataLoader for the MNLI dataset
    device:
        either "cuda" or "cpu"
    tokenizer:
        the pretrained tokenizer
    
    Return
    ------
    all_sources: list of lists
        premise-hypothesis pairs (shape: num_batch x batch_size)
    all_preds: list of lists
        predictions of entailment in premise-hypothesis pairs, which let us know model's prediction
        of premise under each of the 4 fallacies (shape: num_batch x batch_size)
    all_labels: list of lists
        labels of premises with each of the 4 fallacies (shape: num_batch x batch_size)
    """
    with torch.no_grad():
        all_preds = []
        all_sources = []
        all_labels = []

        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y, weights) in enumerate(test_loader):
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)
            prediction = model(pair_token_ids,
                        token_type_ids=seg_ids,
                        attention_mask=mask_ids)
            all_preds += (1-torch.log_softmax(prediction.logits, dim=1).argmax(dim=1)).reshape(-1,4).tolist()
            seqs = tokenizer.batch_decode(pair_token_ids)
            # reshape seqs to (-1,4)
            seqs = [seqs[i:i+4] for i in range(0, len(seqs), 4)]
            all_sources += [[[s.split("[SEP]")[0].replace("[CLS]",""), s.split("[SEP]")[1]] for s in seq] for seq in seqs]
            all_labels += (1-labels).reshape(-1,4).tolist()
        return all_sources, all_preds, all_labels

def prediction_stats(all_sources, all_preds, all_labels):
    """
    Get statistics of predictions of premises with each of the 4 fallacies
    
    Parameters
    ----------
    all_sources : list[list]
        premise-hypothesis pairs - shape: (num_premises x num_fallacies)
    all_preds : list[list]
        predictions of entailment in premise-hypothesis pair - shape: (num_premises x num_fallacies)
    all_labels : list[list]
        labels of premises with each of the 4 fallac - shape: (num_premises x num_fallacies)

    Return
    ------
    df: pandas DataFrame
        Reconstruct to a dataFrame with columns: `PREMISE`, `ORIGINAL_LABEL`, `PREDICTED_E`, `PREDICTED_D`,
        `PREDICTED_R`, `PREDICTED_I`, which looks something like this:
            PREMISE       ORIGINAL_LABEL         PREDICTED_E    PREDICTED_D     PREDICTED_R     PREDICTED_I
            prem_1_Fe     E (apeal to emotion)       1              0               0               0
            prem_n_Fe     E (apeal to emotion)       1              0               1               0  
            ...   
            prem_1_Fd     D (false dilemma)          0              1               0               0
            prem_n_Fd     D (false dilemma)          0              1               1               1
            ...
            prem_1_Fr     R (fallacy of relevance)   0              0               1               0
            prem_n_Fr     R (fallacy of relevance)   0              0               1               0
            ...
            prem_1_Fi     I (intentional)            1              0               0               1
            prem_n_Fi     I (intentional)            1              0               1               1
    """
    "emotion", "dilemmas", "relevancies", "intentions"
    stat = []
    logicpolitics = []
    invalid = []
    for pair, pred, label in zip(all_sources, all_preds, all_labels):
        fallacies = []
        premise = pair[0][0]
        premise = premise.replace("“", "")
        premise = premise.replace("”", "")
        premise = premise.replace("’", "")
        for p in pair:
            if "emotion" in p[1]:
                fallacies.append("E")
            elif "dilemma" in p[1]:
                fallacies.append("D")
            elif "relevance" in p[1]:
                fallacies.append("R")
            else:
                fallacies.append("I")
        assert fallacies == ["E", "D", "R", "I"]
        original_label = fallacies[label.index(1)]
        stat.append([premise, original_label, *pred])
        cut_down = ["# ", "[", "(", "fallacy", "false dilemma", "appeal to emotion"]
        if 1 not in pred or any([c in pair[0][0] for c in cut_down]):
            invalid.append([premise, original_label, *pred])
            continue
        if premise not in [p[0] for p in logicpolitics]:
            logicpolitics.append([premise, original_label, *pred])
    df = pd.DataFrame(stat, columns=["PREMISE", "ORIGINAL_LABEL", "PREDICTED_E", "PREDICTED_D", "PREDICTED_R", "PREDICTED_I"])
    df_lp = pd.DataFrame(logicpolitics, columns=["PREMISE", "ORIGINAL_LABEL", "PREDICTED_E", "PREDICTED_D", "PREDICTED_R", "PREDICTED_I"])
    df_invalid = pd.DataFrame(invalid, columns=["PREMISE", "ORIGINAL_LABEL", "PREDICTED_E", "PREDICTED_D", "PREDICTED_R", "PREDICTED_I"])
    return df, df_lp, df_invalid

def generate_logicpolitics_raw(df_names):
    """
    Append all the dataframes in df_names into one, then convert to raw logicpolitics csv of
    two columns: `source_article`, `updated_label`. The dataframes from `df_names` looks something
    like this (excluding p_F_errors columns):
        SOURCE      P_EMOTION   P_DILEMMA   P_RELEVANCY P_INTENTION
        source_1    Fe          Fd          Fr          Fi
        source_2    Fe          Fd          Fr          Fi
        ...
        source_n    Fe          Fd          Fr          Fi

    Parameters
    ----------
    df_names: list of strings
        list of names of the csv files that contain the premises

    Return
    ------
    df: pandas DataFrame
        DataFrame with two columns: `source_article`, `updated_label`. It looks something like this:
            SOURCE_ARTICLE      UPDATED_LABEL
            source_1_Fe          apeal to emotion
            ...
            source_n_Fe          apeal to emotion
            sourc_1_Fd           false dilemma
            ...
            sourc_n_Fd           false dilemma
            sourc_1_Fr           fallacy of relevance
            ...
            sourc_n_Fr           fallacy of relevance
            sourc_1_Fi           intentional
            ...
            sourc_n_Fi           intentional
    """
    dfs = [pd.read_csv(f"Data/premises/{name}.csv") for name in df_names]

    # Concatenate the DataFrames, ignoring indices
    dfs = pd.concat(dfs, ignore_index=True)
    # Extract columns ["p_emotion", "p_dilemma", "p_relevancy", "p_intention"]
    dfs = dfs[["p_emotion", "p_dilemma", "p_relevancy", "p_intention"]]

    frame = []
    # stack the 4 colmns above into one column, and create a new column with the fallacy name
    for i in dfs.columns:
        frame.append(pd.DataFrame({"source_article": dfs[i].tolist(), "updated_label": [FALLACY_HASH[i]] * len(dfs[i])}))

    df = pd.concat(frame, ignore_index=True)
    df.to_csv("../Data/premises/logicpolitics_raw.csv", index=False)
    return df

if __name__ == "__main__":
    logicpolitics_raw = generate_logicpolitics_raw(["premises1", "premises2", "premises3", "premises4", "premises5"])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_path = "saved_models/electra-logic"
    tokenizer_path = "google/electra-large-discriminator"
    train_strat = 1
    test_strat = 1
    map = "base" # masked-logical-form if using electra-logic-structaware

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.to(device)
    
    fallacy_test = logicpolitics_raw
    # Create MNLI dataset of **ENTAILMENT AND CONTRADICTION** pairs (aka premise entail hypothesis and premise
    # contradicts hypothesis) from each premise and each fallacy
    fallacy_ds = CustomMNLIDataset(tokenizer_path, fallacy_test, fallacy_test, 'updated_label', map, fallacy_test,
                            fallacy=True, train_strat=train_strat, test_strat=test_strat)
    model.resize_token_embeddings(len(fallacy_ds.tokenizer))
    
    # each premise from `fallacy_test` is mapped with each of 4 hypotheses, with entailment/contracdict as labels
    print(fallacy_ds.test_df[:10])

    # Reuse MNLIDataset to create corresponding tokenized MNLI dataset, which is stored in `fallacy_ds.test_data`
    fallacy_ds = CustomMNLIDataset(tokenizer_path, None, None, 'updated_label', map, fallacy_ds.test_df,
                            fallacy=False, train_strat=train_strat, test_strat=test_strat)

    # Create DataLoader for the MNLI dataset, setting batch size to the number of unique labels (no shuffle is done so each batch
    # is 4 premise-hypothesis pairs of the same premise, based on which Electra will predict entailment of the premise with
    # each of the 4 hypotheses)
    test_loader = DataLoader(fallacy_ds.test_data, shuffle=False, batch_size=128)

    all_sources, all_preds, all_labels = filter(model, test_loader, device, fallacy_ds.tokenizer)
    # prediction stats
    df, df_lp, df_invalid = prediction_stats(all_sources, all_preds, all_labels)
    
    df.to_csv("../Data/premises/logicpolitics_stat.csv", index=False)
    df_lp.to_csv("../Data/premises/logicpolitics_filtered.csv", index=False)
    df_invalid.to_csv("../Data/premises/logicpolitics_invalid.csv", index=False)
    