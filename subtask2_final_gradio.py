import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    AutoModelForTokenClassification,
    pipeline
)
import torch
import os
import seaborn as sns
from matplotlib.colors import to_hex
import html


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class SpanClassifierWithStrictF1:
    def __init__(self, model_name="deepset/gbert-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
                        labels[i] = self.label2id[f'B-{type_label}'] # B-compliment
                    else:
                        labels[i] = self.label2id[f'I-{type_label}'] # I-compliment

        return labels

    def compute_metrics(self, eval_pred):
        """Berechne Strict F1 für Trainer"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Konvertiere Vorhersagen zu Spans
        batch_pred_spans = []
        batch_true_spans = []

        for i, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
            # Evaluation-Daten für dieses Beispiel
            if i < len(self.current_eval_data):
                eval_item = self.current_eval_data[i]
                text = eval_item['text']
                offset_mapping = eval_item['offset_mapping']
                true_spans = eval_item['true_spans']

                # Filtere gültige Vorhersagen (keine Padding-Tokens)
                valid_predictions = []
                valid_offsets = []

                for j, (pred_label, true_label) in enumerate(zip(pred_seq, label_seq)):
                    if true_label != -100 and j < len(offset_mapping):
                        valid_predictions.append(pred_label)
                        valid_offsets.append(offset_mapping[j])

                # Konvertiere zu Spans
                pred_spans = self._predictions_to_spans(valid_predictions, valid_offsets, text)
                pred_spans_tuples = [(span['type'], span['start'], span['end']) for span in pred_spans]

                batch_pred_spans.append(pred_spans_tuples)
                batch_true_spans.append(true_spans)

        # Berechne Strict F1
        strict_f1, strict_precision, strict_recall, tp, fp, fn = self._calculate_strict_f1(
            batch_true_spans, batch_pred_spans
        )

        torch.cuda.memory.empty_cache()

        return {
            "strict_f1": torch.tensor(strict_f1),
            "strict_precision": torch.tensor(strict_precision),
            "strict_recall": torch.tensor(strict_recall),
            "true_positives": torch.tensor(tp),
            "false_positives": torch.tensor(fp),
            "false_negatives": torch.tensor(fn)
        }

    def _calculate_strict_f1(self, true_spans_list, pred_spans_list):
        """Berechne Strict F1 über alle Kommentare"""
        tp, fp, fn = 0, 0, 0

        for true_spans, pred_spans in zip(true_spans_list, pred_spans_list):
            # Finde exakte Matches (Typ und Span müssen übereinstimmen)
            matches = self._find_exact_matches(true_spans, pred_spans)

            tp += len(matches)
            fp += len(pred_spans) - len(matches)
            fn += len(true_spans) - len(matches)

        # Berechne Metriken
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall, tp, fp, fn

    def _find_exact_matches(self, true_spans, pred_spans):
        """Finde exakte Matches zwischen True und Predicted Spans"""
        matches = []
        used_pred = set()

        for true_span in true_spans:
            for i, pred_span in enumerate(pred_spans):
                if i not in used_pred and true_span == pred_span:
                    matches.append((true_span, pred_span))
                    used_pred.add(i)
                    break

        return matches

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

    def evaluate_strict_f1(self, comments_df, spans_df):
        """Evaluiere Strict F1 auf Test-Daten"""
        if not hasattr(self, 'model'):
            raise ValueError("Modell muss erst trainiert werden!")

        print("Evaluiere Strict F1...")

        # Vorhersagen für alle Kommentare
        texts = comments_df['comment'].tolist()
        predictions = self.predict(texts)

        # Organisiere True Spans
        spans_grouped = spans_df.groupby(['document', 'comment_id'])
        true_spans_dict = {}
        pred_spans_dict = {}

        for i, (_, row) in enumerate(comments_df.iterrows()):
            key = (row['document'], row['comment_id'])

            # True spans
            if key in spans_grouped.groups:
                true_spans = [(span_type, int(start), int(end))
                              for span_type, start, end in
                              spans_grouped.get_group(key)[['type', 'start', 'end']].values]
            else:
                true_spans = []

            # Predicted spans
            pred_spans = [(span['type'], span['start'], span['end'])
                          for span in predictions[i]['spans']]

            true_spans_dict[key] = true_spans
            pred_spans_dict[key] = pred_spans

        # Berechne Strict F1
        all_true_spans = list(true_spans_dict.values())
        all_pred_spans = list(pred_spans_dict.values())

        f1, precision, recall, tp, fp, fn = self._calculate_strict_f1(all_true_spans, all_pred_spans)

        print(f"\nStrict F1 Ergebnisse:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

        return {
            'strict_f1': f1,
            'strict_precision': precision,
            'strict_recall': recall,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

def convert_spans(row):
    spans = row['predicted_spans']
    document = row['document']
    comment_id = row['comment_id']
    return [{'document': document, 'comment_id': comment_id, 'type': span['type'], 'start': span['start'], 'end': span['end']} for span in spans]

def pred_to_spans(row):
    predicted_labels, offset_mapping, text = row['predicted_labels'], row['offset_mapping'], row['comment']
    return [classifier._predictions_to_spans(predicted_labels, offset_mapping, text)]


def create_highlighted_html(text, spans):
    """Erstelle HTML mit hervorgehobenen Spans"""
    if not spans:
        return html.escape(text)

    # Definiere Farben für verschiedene Span-Typen
    colors = {
        'positive feedback': '#FFE5E5',
        'compliment': '#E5F3FF',
        'affection declaration': '#FFE5F3',
        'encouragement': '#E5FFE5',
        'gratitude': '#FFF5E5',
        'agreement': '#F0E5FF',
        'ambiguous': '#E5E5E5',
        'implicit': '#E5FFFF',
        'group membership': '#FFFFE5',
        'sympathy': '#F5E5FF'
    }

    colors = {
        'positive feedback': '#8dd3c7',  # tealfarbenes Pastell
        'compliment': '#ffffb3',  # helles Pastellgelb
        'affection declaration': '#bebada',  # fliederfarbenes Pastell
        'encouragement': '#fb8072',  # lachsfarbenes Pastell
        'gratitude': '#80b1d3',  # himmelblaues Pastell
        'agreement': '#fdb462',  # pfirsichfarbenes Pastell
        'ambiguous': '#d9d9d9',  # neutrales Pastellgrau
        'implicit': '#fccde5',  # roséfarbenes Pastell
        'group membership': '#b3de69',  # lindgrünes Pastell
        'sympathy': '#bc80bd'  # lavendelfarbenes Pastell
    }

    # Sortiere Spans nach Start-Position
    sorted_spans = sorted(spans, key=lambda x: x['start'])

    html_parts = []
    last_end = 0

    for span in sorted_spans:
        # Text vor dem Span
        if span['start'] > last_end:
            html_parts.append(html.escape(text[last_end:span['start']]))

        # Hervorgehobener Span
        color = colors.get(span['type'], '#EEEEEE')
        span_text = html.escape(text[span['start']:span['end']])
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px; display: inline-block;" title="{span["type"]}">{span_text}</span>')

        last_end = span['end']

    # Restlicher Text
    if last_end < len(text):
        html_parts.append(html.escape(text[last_end:]))

    return ''.join(html_parts)


def create_legend():
    """Erstelle eine Legende für die Span-Typen"""
    #colors = {
    #    'positive feedback': '#FFE5E5',
    #    'compliment': '#E5F3FF',
    #    'affection declaration': '#FFE5F3',
    #    'encouragement': '#E5FFE5',
    #    'gratitude': '#FFF5E5',
    #    'agreement': '#F0E5FF',
    #    'ambiguous': '#E5E5E5',
    #    'implicit': '#E5FFFF',
    #    'group membership': '#FFFFE5',
    #    'sympathy': '#F5E5FF'
    #}

    colors = {
        'positive feedback': '#8dd3c7',  # tealfarbenes Pastell
        'compliment': '#ffffb3',  # helles Pastellgelb
        'affection declaration': '#bebada',  # fliederfarbenes Pastell
        'encouragement': '#fb8072',  # lachsfarbenes Pastell
        'gratitude': '#80b1d3',  # himmelblaues Pastell
        'agreement': '#fdb462',  # pfirsichfarbenes Pastell
        'ambiguous': '#d9d9d9',  # neutrales Pastellgrau
        'implicit': '#fccde5',  # roséfarbenes Pastell
        'group membership': '#b3de69',  # lindgrünes Pastell
        'sympathy': '#bc80bd'  # lavendelfarbenes Pastell
    }
    legend_html = "<div style='margin: 10px 0;'><h4>Candy Speech Types:</h4>"
    for span_type, color in colors.items():
        legend_html += f'<span style="background-color: {color}; padding: 4px 8px; border-radius: 3px; margin: 2px; display: inline-block;">{span_type}</span>'
    legend_html += "</div>"

    return legend_html


def analyze_text(text):
    """Analysiere Text und gebe Ergebnisse zurück"""
    if not text.strip():
        return "Bitte geben Sie einen Text ein.", "", ""

    try:
        # Vorhersage mit dem Classifier
        predictions = classifier.predict([text])
        spans = predictions[0]['spans']

        # Erstelle HTML mit hervorgehobenen Spans
        highlighted_html = create_highlighted_html(text, spans)

        # Erstelle Zusammenfassung
        summary = create_summary(spans)

        # Erstelle detaillierte Span-Informationen
        details = create_details(spans, text)

        return highlighted_html, summary, details

    except Exception as e:
        return f"Fehler bei der Analyse: {str(e)}", "", ""


def create_summary(spans):
    """Erstelle eine Zusammenfassung der gefundenen Spans"""
    if not spans:
        return "Keine Spans gefunden."

    return ""

    span_counts = {}
    for span in spans:
        span_type = span['type']
        span_counts[span_type] = span_counts.get(span_type, 0) + 1

    summary_lines = [f"**Insgesamt {len(spans)} Spans gefunden:**"]
    for span_type, count in sorted(span_counts.items()):
        summary_lines.append(f"- {span_type}: {count}")

    return "\n".join(summary_lines)


def create_details(spans, text):
    """Erstelle detaillierte Informationen über die Spans"""
    if not spans:
        return "Keine Details verfügbar."

    details_lines = ["**Span-Informationen:**"]
    for i, span in enumerate(spans, 1):
        span_text = text[span['start']:span['end']]
        details_lines.append(f"{i}. **{span['type']}** ({span['start']}-{span['end']}): \"{span_text}\"")

    return "\n".join(details_lines)


def load_example_texts():
    """Lade Beispieltexte für die Demo"""
    examples = [
        "Ich stimme allen zu die denken das Roman und Heiko super sind !!!!",
        "da geb ich dir recht ich stehe dir bei die sind einfach nur geil !",
        "OMG, ihr seid einfach der absolute Hammer! 🤩 Eure Videos bringen mich jedes Mal zum Lachen und geben mir so viel Motivation – eure Stimmen klingen mega, eure Parodien sind lustiger als das Original und ihr seht dabei unfassbar toll aus! 😂👌 Bitte macht weiter so! ❤️🎉",
        "Das ist ein wirklich toller Beitrag! Vielen Dank für diese hilfreichen Informationen.",
        "Du bist so klug und hilfreich. Ich bin dir sehr dankbar für deine Unterstützung.",
        "Großartige Arbeit! Das motiviert mich wirklich weiterzumachen.",
        "Das tut mir leid zu hören. Ich hoffe, es wird bald besser für dich.",
    ]
    return examples


# Erstelle die Gradio-Interface
def create_gradio_interface():
    """Erstelle die Gradio-Benutzeroberfläche"""

    with gr.Blocks(title="Span Classifier Demo", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1>🍭 Candy Speech Span Classifier</h1>
            <p>Analysieren Sie Texte und identifizieren Sie verschiedene Arten positiver Kommunikation.</p>
        </div>
        """)

        # Legende
        gr.HTML(create_legend())

        with gr.Row():
            with gr.Column(scale=2):
                # Input
                text_input = gr.Textbox(
                    label="Text eingeben",
                    placeholder="Geben Sie hier den Text ein, den Sie analysieren möchten...",
                    lines=5
                )

                # Buttons
                with gr.Row():
                    analyze_btn = gr.Button("Analysieren", variant="primary")
                    clear_btn = gr.Button("Löschen", variant="secondary")

                # Beispiele
                gr.Examples(
                    examples=load_example_texts(),
                    inputs=text_input,
                    label="Beispieltexte"
                )

                gr.Examples(
                    examples=[ "Bin wegen dir vegan geworden DANKE🫶 Du bist einzigartig und mach bitte weiter 🤍 🧚‍♀️",
                        "Danke für deine tolle Arbeit, auch schön, dass du den Permazidbegriff so wunderbar verwendest <3 Das hast du wirklich alles exzellent gemacht!",
                        "Rafaella Raab ist eine Ikone! Wir sollten alle mehr Tierrechtsaktivismus machen. Höchster Respekt!",
                    ],
                    inputs=text_input,
                    label="Out-of-Distribution Examples (Rafaella Raab)",
                )

                gr.Examples(
                    examples=[
                        "Tolles Video! Hab es einfach stumm geschaltet und tatsächlich eine gute Zeit gehabt.", #aderserial
                        "Auf lautlos ballert der Track noch geiler. 🙏🏻",
                    ],
                    inputs=text_input,
                    label="Adversarial Example (Sarcasm)"
                )

            with gr.Column(scale=2):
                # Outputs
                highlighted_output = gr.HTML(
                    label="Analysierter Text",
                    show_label=True
                )

                summary_output = gr.Markdown(
                    label="Zusammenfassung",
                    show_label=True
                )

                details_output = gr.Markdown(
                    label="Details",
                    show_label=True
                )

        # Info-Bereich
        with gr.Accordion("ℹ️ Informationen zum Modell", open=False):
            gr.Markdown("""
            ### Über dieses Modell

            Dieses Modell identifiziert verschiedene Arten positiver Kommunikation in Texten:

            - **Positive Feedback**: Allgemein positive Rückmeldungen
            - **Compliment**: Direkte Komplimente
            - **Affection Declaration**: Liebesbekundungen oder Zuneigung
            - **Encouragement**: Ermutigung und Motivation
            - **Gratitude**: Dankbarkeit und Wertschätzung
            - **Agreement**: Zustimmung und Einverständnis
            - **Ambiguous**: Mehrdeutige positive Aussagen
            - **Implicit**: Implizite positive Kommunikation
            - **Group Membership**: Zugehörigkeitsgefühl
            - **Sympathy**: Mitgefühl und Empathie

            ### Verwendung
            1. Geben Sie einen Text in das Eingabefeld ein
            2. Klicken Sie auf "Analysieren"
            3. Betrachten Sie die hervorgehobenen Spans im analysierten Text
            4. Überprüfen Sie die Zusammenfassung und Details
            """)

        # Event-Handler
        analyze_btn.click(
            fn=analyze_text,
            inputs=text_input,
            outputs=[highlighted_output, summary_output, details_output]
        )

        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[text_input, highlighted_output, summary_output, details_output]
        )

        # Auto-Analyse bei Beispiel-Auswahl
        text_input.change(
            fn=analyze_text,
            inputs=text_input,
            outputs=[highlighted_output, summary_output, details_output]
        )

    return demo



if __name__ == "__main__":
    classifier = SpanClassifierWithStrictF1('xlm-roberta-large')

    classifier.model = AutoModelForTokenClassification.from_pretrained(
        'xlm-roberta-large',
        num_labels=len(classifier.labels),
        id2label=classifier.id2label,
        label2id=classifier.label2id
    )
    classifier.model.load_state_dict(torch.load('./experiments/exp027/exp027-2_retraining_final_model.pth'))
    classifier.model.eval()

    print("Modell geladen! Starte Gradio-Interface...")

    # Erstelle und starte die Demo
    demo = create_gradio_interface()

    # Starte die Demo
    demo.launch(
        server_name="0.0.0.0",  # Für externen Zugriff
        server_port=7860,
        debug=True,
        show_error=True
    )