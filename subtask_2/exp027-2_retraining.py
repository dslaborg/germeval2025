import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from multiset import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.environ["WANDB_PROJECT"]="GermEval2025-Substask2"
os.environ["WANDB_LOG_MODEL"]="false"

experiment_name = 'exp027-2_retraining'

ALL_LABELS = ["affection declaration","agreement","ambiguous",
              "compliment","encouragement","gratitude","group membership",
              "implicit","positive feedback","sympathy"]


def fine_grained_flausch_by_label(gold, predicted):
    gold['cid']= gold['document']+"_"+gold['comment_id'].apply(str)
    predicted['cid']= predicted['document']+"_"+predicted['comment_id'].apply(str)

    # annotation sets (predicted)
    pred_spans = Multiset()
    pred_spans_loose = Multiset()
    pred_types = Multiset()

    # annotation sets (gold)
    gold_spans = Multiset()
    gold_spans_loose = Multiset()
    gold_types = Multiset()

    for row in predicted.itertuples(index=False):
        pred_spans.add((row.cid,row.type,row.start,row.end))
        pred_spans_loose.add((row.cid,row.start,row.end))
        pred_types.add((row.cid,row.type))
    for row in gold.itertuples(index=False):
        gold_spans.add((row.cid,row.type,row.start,row.end))
        gold_spans_loose.add((row.cid,row.start,row.end))
        gold_types.add((row.cid,row.type))

    # precision = true_pos / true_pos + false_pos
    # recall = true_pos / true_pos + false_neg
    # f_1 = 2 * prec * rec / (prec + rec)

    results = {'TOTAL': {'STRICT': {},'SPANS': {},'TYPES': {}}}
    # label-wise evaluation (only for strict and type)
    for label in ALL_LABELS:
        results[label] = {'STRICT': {},'TYPES': {}}
        gold_spans_x = set(filter(lambda x: x[1].__eq__(label), gold_spans))
        pred_spans_x = set(filter(lambda x: x[1].__eq__(label), pred_spans))
        gold_types_x = set(filter(lambda x: x[1].__eq__(label), gold_types))
        pred_types_x = set(filter(lambda x: x[1].__eq__(label), pred_types))

        # strict: spans + type must match
        ### NOTE: x and y / x returns 0 if x = 0 and y/x otherwise (test for zero division)
        strict_p = float(len(pred_spans_x)) and float( len(gold_spans_x.intersection(pred_spans_x))) / len(pred_spans_x)
        strict_r = float(len(gold_spans_x)) and float( len(gold_spans_x.intersection(pred_spans_x))) / len(gold_spans_x)
        strict_f = (strict_p + strict_r) and 2 * strict_p * strict_r / (strict_p + strict_r)
        results[label]['STRICT']['prec'] = strict_p
        results[label]['STRICT']['rec'] = strict_r
        results[label]['STRICT']['f1'] = strict_f

        # detection mode: only types must match (per post)
        types_p = float(len(pred_types_x)) and float( len(gold_types_x.intersection(pred_types_x))) / len(pred_types_x)
        types_r = float(len(gold_types_x)) and float( len(gold_types_x.intersection(pred_types_x))) / len(gold_types_x)
        types_f = (types_p + types_r) and 2 * types_p * types_r / (types_p + types_r)
        results[label]['TYPES']['prec'] = types_p
        results[label]['TYPES']['rec'] = types_r
        results[label]['TYPES']['f1'] = types_f

    # Overall evaluation
    # strict: spans + type must match
    strict_p = float(len(pred_spans)) and float( len(gold_spans.intersection(pred_spans))) / len(pred_spans)
    strict_r = float(len(gold_spans)) and float( len(gold_spans.intersection(pred_spans))) / len(gold_spans)
    strict_f = (strict_p + strict_r) and 2 * strict_p * strict_r / (strict_p + strict_r)
    results['TOTAL']['STRICT']['prec'] = strict_p
    results['TOTAL']['STRICT']['rec'] = strict_r
    results['TOTAL']['STRICT']['f1'] = strict_f

    # spans: spans must match
    spans_p = float(len(pred_spans_loose)) and float( len(gold_spans_loose.intersection(pred_spans_loose))) / len(pred_spans_loose)
    spans_r = float(len(gold_spans_loose)) and float( len(gold_spans_loose.intersection(pred_spans_loose))) / len(gold_spans_loose)
    spans_f = (spans_p + spans_r) and 2 * spans_p * spans_r / (spans_p + spans_r)
    results['TOTAL']['SPANS']['prec'] = spans_p
    results['TOTAL']['SPANS']['rec'] = spans_r
    results['TOTAL']['SPANS']['f1'] = spans_f

    # detection mode: only types must match (per post)
    types_p = float(len(pred_types)) and float( len(gold_types.intersection(pred_types))) / len(pred_types)
    types_r = float(len(gold_types)) and float( len(gold_types.intersection(pred_types))) / len(gold_types)
    types_f = (types_p + types_r) and 2 * types_p * types_r / (types_p + types_r)
    results['TOTAL']['TYPES']['prec'] = types_p
    results['TOTAL']['TYPES']['rec'] = types_r
    results['TOTAL']['TYPES']['f1'] = types_f

    return results

class SpanClassifierWithStrictF1:
    def __init__(self, model_name="deepset/gbert-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

        self.labels =[
            "O",
            "B-positive feedback", "B-compliment", "B-affection declaration", "B-encouragement", "B-gratitude", "B-agreement", "B-ambiguous", "B-implicit", "B-group membership", "B-sympathy",
            "I-positive feedback", "I-compliment", "I-affection declaration", "I-encouragement", "I-gratitude", "I-agreement", "I-ambiguous", "I-implicit", "I-group membership", "I-sympathy"
        ]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def create_dataset(self, comments_df, spans_df):
        """Erstelle Dataset mit BIO-Labels und speichere Evaluation-Daten"""
        examples = []
        eval_data = []  # Für Strict F1 Berechnung

        spans_grouped = spans_df.groupby(['document', 'comment_id'])

        for _, row in comments_df.iterrows():
            text = row['comment']
            document = row['document']
            comment_id = row['comment_id']
            key = (document, comment_id)

            # True spans für diesen Kommentar
            if key in spans_grouped.groups:
                true_spans = [(span_type, int(start), int(end))
                              for span_type, start, end in
                              spans_grouped.get_group(key)[['type', 'start', 'end']].values]
            else:
                true_spans = []

            # Tokenisierung
            tokenized = self.tokenizer(text, truncation=True, max_length=512,
                                       return_offsets_mapping=True)

            # BIO-Labels erstellen
            labels = self._create_bio_labels(tokenized['offset_mapping'],
                                             spans_grouped.get_group(key)[['start', 'end', 'type']].values
                                             if key in spans_grouped.groups else [])

            examples.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            })

            # Evaluation-Daten speichern
            eval_data.append({
                'text': text,
                'offset_mapping': tokenized['offset_mapping'],
                'true_spans': true_spans,
                'document': document,
                'comment_id': comment_id
            })

        return examples, eval_data

    def _create_bio_labels(self, offset_mapping, spans):
        """Erstelle BIO-Labels für Tokens"""
        labels = [0] * len(offset_mapping)  # 0 = "O"

        for start, end, type_label in spans:
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start is None:  # Spezielle Tokens
                    continue

                # Token überlappt mit Span
                if token_start < end and token_end > start:
                    if token_start <= start:
                        if labels[i] != 0:
                            # dont overwrite labels if spans are overlapping; just skip the span
                            break
                        labels[i] = self.label2id[f'B-{type_label}'] # B-compliment
                    else:
                        labels[i] = self.label2id[f'I-{type_label}'] # I-compliment

        return labels

    def _predictions_to_dataframe(self, predictions_list, comments_df_subset):
        """Konvertiere Vorhersagen zu DataFrame für Flausch-Metrik"""
        pred_data = []

        for i, pred in enumerate(predictions_list):
            if i < len(comments_df_subset):
                row = comments_df_subset.iloc[i]
                document = row['document']
                comment_id = row['comment_id']

                for span in pred['spans']:
                    pred_data.append({
                        'document': document,
                        'comment_id': comment_id,
                        'type': span['type'],
                        'start': span['start'],
                        'end': span['end']
                    })

        return pd.DataFrame(pred_data)

    # --- helper that builds a DataFrame of spans from eval data + predictions ---
    def _build_span_dfs(self, eval_data, batch_pred_spans):
        """
        eval_data: list of dicts with keys document, comment_id, true_spans
        batch_pred_spans: list of lists of (type, start, end)
        returns (gold_df, pred_df) suitable for fine_grained_flausch_by_label
        """
        rows_gold = []
        rows_pred = []
        for item, pred_spans in zip(eval_data, batch_pred_spans):
            doc = item['document']
            cid = item['comment_id']
            # gold
            for t, s, e in item['true_spans']:
                rows_gold.append({
                    'document': doc,
                    'comment_id': cid,
                    'type': t,
                    'start': s,
                    'end':   e
                })
            # pred
            for t, s, e in pred_spans:
                rows_pred.append({
                    'document': doc,
                    'comment_id': cid,
                    'type': t,
                    'start': s,
                    'end':   e
                })
        gold_df = pd.DataFrame(rows_gold, columns=['document','comment_id','type','start','end'])
        pred_df = pd.DataFrame(rows_pred, columns=['document','comment_id','type','start','end'])
        return gold_df, pred_df


    def compute_metrics(self, eval_pred):
        """
        Called by the HF-Trainer at each evaluation step.
        We collect batch predictions, reconstruct gold/pred spans,
        call fine_grained_flausch_by_label and return the TOTAL/STRICT metrics.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)

        # reconstruct spans per example in this batch
        batch_pred_spans = []
        for i, (p_seq, lab_seq) in enumerate(zip(preds, labels)):
            # skip padding (-100)
            valid_preds = []
            valid_offsets = []
            offsets = self.current_eval_data[i]['offset_mapping']
            for j,(p,l) in enumerate(zip(p_seq, lab_seq)):
                if l != -100:
                    valid_preds.append(int(p))
                    valid_offsets.append(offsets[j])
            # convert to spans
            pred_spans = self._predictions_to_spans(valid_preds, valid_offsets,
                                                    self.current_eval_data[i]['text'])
            # to (type, start, end)-tuples
            batch_pred_spans.append([(sp['type'], sp['start'], sp['end'])
                                     for sp in pred_spans])

        # build the gold/pred DataFrames
        gold_df, pred_df = self._build_span_dfs(self.current_eval_data,
                                                batch_pred_spans)

        # call your fine-grained metrics
        results = fine_grained_flausch_by_label(gold_df, pred_df)

        # extract the TOTAL/STRICT metrics
        total = results['TOTAL']['STRICT']
        return {
            'strict_prec': torch.tensor(total['prec'], dtype=torch.float32),
            'strict_rec':  torch.tensor(total['rec'],  dtype=torch.float32),
            'strict_f1':   torch.tensor(total['f1'],   dtype=torch.float32),
        }


    def evaluate_by_label(self, comments_df, spans_df):
        """
        Replace evaluate_strict_f1. Runs a full pass over all comments,
        uses self.predict() to get spans, then calls your fine_grained_flausch_by_label
        and prints & returns the TOTAL metrics.
        """
        # 1) run predictions
        texts = comments_df['comment'].tolist()
        docs =  comments_df['document'].tolist()
        cids =  comments_df['comment_id'].tolist()
        preds = self.predict(texts)

        # 2) build gold and pred lists
        gold_rows = []
        for (_, row) in comments_df.iterrows():
            key = (row['document'], row['comment_id'])
            # get all true spans for this comment_id
            group = spans_df[
                (spans_df.document==row['document']) &
                (spans_df.comment_id==row['comment_id'])
            ]
            for _, sp in group.iterrows():
                gold_rows.append({
                    'document': row['document'],
                    'comment_id': row['comment_id'],
                    'type': sp['type'],
                    'start': sp['start'],
                    'end': sp['end']
                })

        pred_rows = []
        for doc, cid, p in zip(docs, cids, preds):
            for sp in p['spans']:
                pred_rows.append({
                    'document': doc,
                    'comment_id': cid,
                    'type': sp['type'],
                    'start': sp['start'],
                    'end': sp['end']
                })

        gold_df = pd.DataFrame(gold_rows, columns=['document','comment_id','type','start','end'])
        pred_df = pd.DataFrame(pred_rows, columns=['document','comment_id','type','start','end'])

        # 3) call fine-grained
        results = fine_grained_flausch_by_label(gold_df, pred_df)

        # 4) extract and print
        total = results['TOTAL']
        print("\n=== EVALUATION BY FLAUSCH METRICS ===")
        for mode in ['STRICT','SPANS','TYPES']:
            m = total[mode]
            print(f"{mode:6}  P={m['prec']:.4f}  R={m['rec']:.4f}  F1={m['f1']:.4f}")

        return results

    def _predictions_to_spans(self, predicted_labels, offset_mapping, text):
        """Konvertiere Token-Vorhersagen zu Spans"""
        spans = []
        current_span = None

        for i, label_id in enumerate(predicted_labels):
            if i >= len(offset_mapping):
                break

            label = self.id2label[label_id]
            token_start, token_end = offset_mapping[i]

            if token_start is None:
                continue

            if label.startswith('B-'):
                if current_span:
                    spans.append(current_span)
                current_span = {
                    'type': label[2:],
                    'start': token_start,
                    'end': token_end,
                    'text': text[token_start:token_end]
                }
            elif label.startswith('I-') and current_span:
                current_span['end'] = token_end
                current_span['text'] = text[current_span['start']:current_span['end']]
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None

        if current_span:
            spans.append(current_span)

        return spans

    def predict(self, texts):
        """Vorhersage für neue Texte"""
        if not hasattr(self, 'model'):
            raise ValueError("Modell muss erst trainiert werden!")

        predictions = []
        device = next(self.model.parameters()).device

        for text in texts:
            # Tokenisierung
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=512, return_offsets_mapping=True)

            offset_mapping = inputs.pop('offset_mapping')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Vorhersage
            with torch.no_grad():
                outputs = self.model(**inputs)

            predicted_labels = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

            # Spans extrahieren
            spans = self._predictions_to_spans(predicted_labels, offset_mapping[0], text)
            predictions.append({'text': text, 'spans': spans})

        return predictions

    def train(self, comments_df, spans_df, experiment_name):
        wandb.init(project=os.environ["WANDB_PROJECT"], name=f"{experiment_name}",
                   group=experiment_name)


        # Dataset neu erstellen für diesen Fold
        examples, eval_data = self.create_dataset(comments_df, spans_df)
        train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)

        # Evaluation-Daten entsprechend aufteilen
        train_indices, val_indices = train_test_split(range(len(examples)), test_size=0.1, random_state=42)
        self.current_eval_data = [eval_data[i] for i in val_indices]

        test_comments = comments_df.iloc[val_indices].reset_index(drop=True)

        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)

        # Modell neu initialisieren
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # Training-Argumente
        fold_output_dir = f"{experiment_name}"
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            learning_rate=2e-5,
            warmup_steps=500,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=20,
            eval_strategy="steps",
            eval_steps=40,
            save_strategy="steps",
            save_steps=40,
            load_best_model_at_end=True,
            metric_for_best_model="strict_f1",
            greater_is_better=True,
            logging_steps=10,
            logging_strategy="steps",
            report_to="all",
            disable_tqdm=False,
            seed=42,
            save_total_limit=3,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=87)]
            # 87 steps = 3.0 epochs with 29 steps per epoch
        )

        # Training
        print(f"Training auf {len(train_dataset)} Beispielen")
        print(f"Validation auf {len(val_dataset)} Beispielen")
        trainer.train()

        # Aktuelles Modell speichern
        self.model = model

        # Modell evaluieren auf Test-Daten
        print(f"Evaluierung auf {len(test_comments)} Test-Beispielen")
        metrics = self.evaluate_by_label(test_comments, spans_df)
        wandb.log({
            'strict_f1': metrics['TOTAL']['STRICT']['f1'],
            'strict_precision': metrics['TOTAL']['STRICT']['prec'],
            'strict_recall': metrics['TOTAL']['STRICT']['rec'],
            'spans_f1': metrics['TOTAL']['SPANS']['f1'],
            'types_f1': metrics['TOTAL']['TYPES']['f1']
        })


        # Speichere Modell
        torch.save(model.state_dict(), f'{fold_output_dir}_model.pth')

        torch.cuda.memory.empty_cache()
        wandb.finish()

        return trainer


    def cross_validate(self, comments_df, spans_df, n_splits=5, output_dir_prefix="span-classifier-cv"):
        """Führe n-fache Kreuzvalidierung mit StratifiedKFold durch"""

        # Erstelle Label für Stratifizierung (basierend auf dem ersten Span types eines Kommentars)
        strat_labels = []
        spans_grouped = spans_df.groupby(['document', 'comment_id'])
        for _, row in comments_df.iterrows():
            key = (row['document'], row['comment_id'])
            # 1 wenn Kommentar Spans hat, sonst 0
            has_spans = spans_grouped.get_group(key).iloc[0]['type'] if key in spans_grouped.groups and len(spans_grouped.get_group(key)) > 0 else 0
            strat_labels.append(has_spans)

        # Erstelle StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Speichere Metriken für jeden Fold
        fold_metrics = []

        # Iteriere über Folds
        for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(comments_df)), strat_labels)):
            if '--fold' in sys.argv:
                fold_arg = int(sys.argv[sys.argv.index('--fold') + 1])
                if fold + 1 != fold_arg:
                    continue

            wandb.init(project=os.environ["WANDB_PROJECT"], name=f"{experiment_name}-fold-{fold+1}",
                       group=experiment_name)

            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{n_splits}")
            print(f"{'='*50}")

            # Kommentare für diesen Fold
            train_comments = comments_df.iloc[train_idx].reset_index(drop=True)
            test_comments = comments_df.iloc[test_idx].reset_index(drop=True)

            # Dataset neu erstellen für diesen Fold
            examples, eval_data = self.create_dataset(train_comments, spans_df)
            train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)

            # Evaluation-Daten entsprechend aufteilen
            train_indices, val_indices = train_test_split(range(len(examples)), test_size=0.1, random_state=42)
            self.current_eval_data = [eval_data[i] for i in val_indices]

            train_dataset = Dataset.from_list(train_examples)
            val_dataset = Dataset.from_list(val_examples)

            # Modell neu initialisieren
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id
            )

            # Training-Argumente
            fold_output_dir = f"{output_dir_prefix}-fold-{fold+1}"
            training_args = TrainingArguments(
                output_dir=fold_output_dir,
                learning_rate=2e-5,
                warmup_steps=500,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=15,
                eval_strategy="steps",
                eval_steps=40,
                save_strategy="steps",
                save_steps=40,
                load_best_model_at_end=True,
                metric_for_best_model="strict_f1",
                greater_is_better=True,
                logging_steps=10,
                logging_strategy="steps",
                report_to="all",
                disable_tqdm=False,
                seed=42,
                save_total_limit=3,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=DataCollatorForTokenClassification(self.tokenizer),
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=87)] # 87 steps = 3.0 epochs with 29 steps per epoch
            )

            # Training
            print(f"Training auf {len(train_dataset)} Beispielen")
            print(f"Validation auf {len(val_dataset)} Beispielen")
            trainer.train()

            # Aktuelles Modell speichern
            self.model = model

            # Modell evaluieren auf Test-Daten
            print(f"Evaluierung auf {len(test_comments)} Test-Beispielen")
            flausch_results = self.evaluate_by_label(test_comments, spans_df)

            # Extrahiere Hauptmetriken für fold_metrics
            metrics = {
                'strict_f1': flausch_results['TOTAL']['STRICT']['f1'],
                'strict_precision': flausch_results['TOTAL']['STRICT']['prec'],
                'strict_recall': flausch_results['TOTAL']['STRICT']['rec'],
                'spans_f1': flausch_results['TOTAL']['SPANS']['f1'],
                'spans_precision': flausch_results['TOTAL']['SPANS']['prec'],
                'spans_recall': flausch_results['TOTAL']['SPANS']['rec'],
                'types_f1': flausch_results['TOTAL']['TYPES']['f1'],
                'types_precision': flausch_results['TOTAL']['TYPES']['prec'],
                'types_recall': flausch_results['TOTAL']['TYPES']['rec'],
                'full_results': flausch_results
            }

            fold_metrics.append(metrics)
            wandb.log(metrics, step=fold + 1)

            # Speichere Modell
            torch.save(model.state_dict(), f'{fold_output_dir}_model.pth')

            test_predictions = self.predict(test_comments['comment'].tolist())

            # Speichere Metriken
            with open(f"test_results.{experiment_name}.fold-{fold+1}.pkl", "wb") as p:
                pickle.dump((train_comments, test_comments, test_predictions, train_examples, val_examples), p)

            with open(f"scores.{experiment_name}.txt", 'a') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fold {fold+1} Ergebnisse:\n')
                f.write(f"[{experiment_name} fold-{fold+1} {metrics}\n")

            torch.cuda.memory.empty_cache()
            wandb.finish()

        # Zusammenfassung ausgeben
        print("\n" + "="*50)
        print("Kreuzvalidierung abgeschlossen")
        print("="*50)

        # Berechne Durchschnitts-Metriken
        avg_f1 = np.mean([m['strict_f1'] for m in fold_metrics])
        avg_precision = np.mean([m['strict_precision'] for m in fold_metrics])
        avg_recall = np.mean([m['strict_recall'] for m in fold_metrics])

        print(f"\nDurchschnittliche Metriken über {n_splits} Folds:")
        print(f"Precision: {avg_precision:.10f}")
        print(f"Recall:    {avg_recall:.10f}")
        print(f"F1-Score:  {avg_f1:.10f}")

        # Std-Abweichung
        std_f1 = np.std([m['strict_f1'] for m in fold_metrics])
        std_precision = np.std([m['strict_precision'] for m in fold_metrics])
        std_recall = np.std([m['strict_recall'] for m in fold_metrics])

        print(f"\nStandardabweichung über {n_splits} Folds:")
        print(f"Precision: {std_precision:.10f}")
        print(f"Recall:    {std_recall:.10f}")
        print(f"F1-Score:  {std_f1:.10f}")

        # Ergebnisse für jeden Fold ausgeben
        for fold, metrics in enumerate(fold_metrics):
            print(f"\nFold {fold+1} Ergebnisse:")
            print(f"Precision: {metrics['strict_precision']:.4f}")
            print(f"Recall:    {metrics['strict_recall']:.4f}")
            print(f"F1-Score:  {metrics['strict_f1']:.4f}")

        return {
            'fold_metrics': fold_metrics,
            'avg_metrics': {
                'strict_f1': avg_f1,
                'strict_precision': avg_precision,
                'strict_recall': avg_recall
            },
            'std_metrics': {
                'strict_f1': std_f1,
                'strict_precision': std_precision,
                'strict_recall': std_recall
            }
        }



# Daten laden
comments: pd.DataFrame = pd.read_csv("../../share-GermEval2025-data/Data/training data/comments.csv")
task1: pd.DataFrame = pd.read_csv("../../share-GermEval2025-data/Data/training data/task1.csv")
task2: pd.DataFrame = pd.read_csv("../../share-GermEval2025-data/Data/training data/task2.csv")
comments = comments.merge(task1, on=["document", "comment_id"])

test_data: pd.DataFrame = pd.read_csv("../../share-GermEval2025-data/Data/test data/comments.csv")

# Wähle Teilmenge der Daten für Experiment (z.B. 17000 Kommentare)
experiment_data = comments

# Klassifikator mit Strict F1
classifier = SpanClassifierWithStrictF1('xlm-roberta-large')

# 5-fold Cross-Validation durchführen
#cv_results = classifier.cross_validate(
#    experiment_data,
#    task2,
#    n_splits=5,
#    output_dir_prefix=experiment_name
#)
#
## write results to text file
#with open(f"scores.{experiment_name}.txt", 'a') as f:
#    f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] KFold cross validation of {experiment_name}\n')
#    f.write(f'{cv_results}\n')

# Optional: Finales Modell auf allen Daten trainieren
trainer = classifier.train(experiment_data, task2, f'{experiment_name}-final')
torch.save(classifier.model.state_dict(), f'{experiment_name}_final_model.pth')

# Test-Vorhersage mit finalem Modell
test_texts = ["Das ist ein toller Kommentar!", "Schlechter Text hier.",
              "Sehr gutes Video. Danke! Ich finde Dich echt toll!", "Du bist doof!", "Das Licht ist echt gut.",
              "Team Einhorn", "Macht unbedingt weiter so!", "Das sehe ich ganz genauso.", "Stimmt, Du hast vollkommen Recht!",
              "Ich bin so dankbar ein #Lochinator zu sein"]

predictions = classifier.predict(test_texts)

for pred in predictions:
    print(f"\nText: {pred['text']}")
    for span in pred['spans']:
        print(f"  Span: '{span['text']}' ({span['start']}-{span['end']}) - {span['type']}")




