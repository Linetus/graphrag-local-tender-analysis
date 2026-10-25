"""
Evaluación comparativa: LLM sin pipeline vs GraphRAG
Métricas: Token F1, ROUGE-L, BERTScore, RAGAS (AnswerSimilarity + AnswerCorrectness)
"""

import re
import json
import ollama
from rouge_score import rouge_scorer as rouge_lib
from bert_score import score as bert_score_fn

# ---------------------------------------------------------------------------
# Dataset: preguntas, ground truths y respuestas del sistema GraphRAG
# (respuestas extraídas del PDF TFG_Pruebas)
# ---------------------------------------------------------------------------

QUESTIONS = [
    "¿Cuáles son los niveles de peligrosidad de un ciberincidente según el CCN-STIC-817?",
    "¿Qué organismo debe ser notificado ante un incidente de categoría ALTA en la Administración Pública?",
    "¿Cuál es el plazo máximo para reportar un incidente al CCN-CERT una vez detectado?",
    "¿Qué tipo de ataques cubre específicamente la guía CCN-STIC-812?",
    "¿Qué cabecera HTTP recomienda la guía para mitigar ataques XSS?",
    "¿Qué fases componen el ciclo de gestión de una crisis de ciberseguridad según INCIBE?",
    "¿Qué es un CSIRT según la guía de INCIBE?",
    "¿Cuándo se considera que un incidente de seguridad escala a 'crisis'?",
]

GROUND_TRUTHS = [
    "Muy Alto, Alto, Medio, Bajo y Sin Clasificar (5 niveles)",
    "El CCN-CERT (Centro Criptológico Nacional)",
    "El documento no establece un plazo explícito de notificación inicial; menciona 60 días como plazo de cierre del seguimiento por el SAT",
    "Ataques a entornos y aplicaciones web: inyección SQL, XSS, CSRF, entre otros",
    "No aparece explícitamente en la guía",
    "Preparación, Identificación, Respuesta, Cierre",
    "Computer Security Incident Response Team, equipo de respuesta a incidentes de seguridad",
    "Cuando supera la capacidad de respuesta habitual de la organización o afecta a su continuidad de negocio",
]

GRAPHRAG_ANSWERS = [
    "Los niveles de peligrosidad de un ciberincidente según el CCN-STIC-817 son cinco: Bajo, Medio, Alto, Muy Alto y Crítico.",
    "El CCN-CERT (Centro Criptológico Nacional) es el organismo que debe ser notificado ante un incidente de categoría ALTA en la Administración Pública.",
    "Según el documento CCN-STIC-817, el plazo máximo es de 60 días, que es cuando el sistema SAT cierra el incidente sin respuesta.",
    "La guía CCN-STIC-812 cubre ataques de inyección SQL, Cross-Site Scripting (XSS), desplazamiento por directorios, referencias directas a ficheros, ejecución de comandos del sistema operativo y HTTP Response Splitting.",
    "La guía no menciona específicamente ninguna cabecera HTTP para mitigar ataques XSS. Recomienda aplicar defensa en profundidad y tokens anti-CSRF.",
    "Las fases son: 1. Preparación, 2. Identificación y análisis, 3. Respuesta y comunicación, 4. Cierre.",
    "Según la guía de INCIBE, un CSIRT (Computer Security Incident Response Team) es un equipo responsable de gestionar y responder a incidentes de seguridad en computadoras y redes.",
    "Un incidente de seguridad escala a crisis cuando causa daños significativos, riesgo de pérdida de datos importantes, interrupción de servicios críticos o pone en riesgo la reputación de la organización.",
]

QUESTION_IDS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

# ---------------------------------------------------------------------------
# 1. Obtener respuestas baseline (LLM sin contexto)
# ---------------------------------------------------------------------------

def get_baseline_answers(questions: list[str], model: str = "llama3.1:8b") -> list[str]:
    answers = []
    for i, q in enumerate(questions):
        print(f"  Baseline {i+1}/{len(questions)}: {q[:60]}...")
        try:
            resp = ollama.generate(
                model=model,
                prompt=(
                    "Responde de forma breve y precisa a la siguiente pregunta "
                    "sobre ciberseguridad. Si no sabes la respuesta exacta, di "
                    "lo que creas más probable.\n\n"
                    f"Pregunta: {q}\n\nRespuesta:"
                ),
                options={"temperature": 0.0},
            )
            answers.append(resp["response"].strip())
        except Exception as e:
            print(f"    ERROR: {e}")
            answers.append("")
    return answers

# ---------------------------------------------------------------------------
# 2. Token F1
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    return 2 * precision * recall / (precision + recall)

# ---------------------------------------------------------------------------
# 3. ROUGE-L
# ---------------------------------------------------------------------------

_rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)

def rouge_l(prediction: str, reference: str) -> float:
    return _rouge.score(reference, prediction)["rougeL"].fmeasure

# ---------------------------------------------------------------------------
# 4. BERTScore  (modelo multilingüe para español)
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    print("  Calculando BERTScore (modelo multilingüe)...")
    _, _, F1 = bert_score_fn(
        predictions, references,
        model_type="xlm-roberta-base",
        lang="es",
        verbose=False,
        device="cpu",
    )
    return F1.tolist()

# ---------------------------------------------------------------------------
# 5. RAGAS: AnswerSimilarity + AnswerCorrectness (con Ollama local)
# ---------------------------------------------------------------------------

def compute_ragas(
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
    llm_model: str = "llama3.1:8b",
    embed_model: str = "all-MiniLM-L6-v2",
) -> dict:
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import AnswerSimilarity, AnswerCorrectness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("  Configurando RAGAS con Ollama LLM + HuggingFace embeddings...")
        ragas_llm = LangchainLLMWrapper(ChatOllama(model=llm_model, temperature=0))
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=embed_model,
                                  model_kwargs={"device": "cpu"})
        )

        samples = [
            SingleTurnSample(user_input=q, response=a, reference=r)
            for q, a, r in zip(questions, answers, ground_truths)
        ]
        dataset = EvaluationDataset(samples=samples)

        metrics = [AnswerSimilarity(embeddings=ragas_emb),
                   AnswerCorrectness(llm=ragas_llm, embeddings=ragas_emb)]

        result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm,
                          embeddings=ragas_emb, raise_exceptions=False)
        df = result.to_pandas()
        return {
            "answer_similarity": df["answer_similarity"].tolist(),
            "answer_correctness": df["answer_correctness"].tolist(),
        }
    except Exception as e:
        print(f"  RAGAS error: {e}")
        return {"answer_similarity": [None] * len(questions),
                "answer_correctness": [None] * len(questions)}

# ---------------------------------------------------------------------------
# 6. Formatear y mostrar tabla de resultados
# ---------------------------------------------------------------------------

def fmt(v) -> str:
    if v is None:
        return "  —  "
    return f"{v:.3f}"

def print_results_table(label: str, ids, questions, answers, references,
                        f1s, rls, bss, ragas_sim, ragas_cor):
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    header = f"{'ID':<4} {'F1':>6} {'ROUGE-L':>8} {'BERTScore':>10} {'RAGAS-Sim':>10} {'RAGAS-Cor':>10}  Respuesta (primeros 80 chars)"
    print(header)
    print("-" * 100)
    for i, qid in enumerate(ids):
        row = (f"{qid:<4} {fmt(f1s[i]):>6} {fmt(rls[i]):>8} {fmt(bss[i]):>10} "
               f"{fmt(ragas_sim[i]):>10} {fmt(ragas_cor[i]):>10}  {answers[i][:80]}")
        print(row)
    print("-" * 100)
    valid_f1  = [x for x in f1s  if x is not None]
    valid_rl  = [x for x in rls  if x is not None]
    valid_bs  = [x for x in bss  if x is not None]
    valid_rs  = [x for x in ragas_sim if x is not None]
    valid_rc  = [x for x in ragas_cor if x is not None]
    print(f"{'MEDIA':<4} "
          f"{sum(valid_f1)/len(valid_f1):>6.3f} "
          f"{sum(valid_rl)/len(valid_rl):>8.3f} "
          f"{sum(valid_bs)/len(valid_bs):>10.3f} "
          f"{(sum(valid_rs)/len(valid_rs) if valid_rs else 0):>10.3f} "
          f"{(sum(valid_rc)/len(valid_rc) if valid_rc else 0):>10.3f}")
    print(f"{'='*100}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== PASO 1: Generando respuestas baseline (LLM sin contexto) ===")
    baseline_answers = get_baseline_answers(QUESTIONS)

    print("\n=== PASO 2: Métricas clásicas (Token F1, ROUGE-L) ===")
    baseline_f1 = [token_f1(a, r) for a, r in zip(baseline_answers, GROUND_TRUTHS)]
    baseline_rl = [rouge_l(a, r)  for a, r in zip(baseline_answers, GROUND_TRUTHS)]
    graphrag_f1 = [token_f1(a, r) for a, r in zip(GRAPHRAG_ANSWERS, GROUND_TRUTHS)]
    graphrag_rl = [rouge_l(a, r)  for a, r in zip(GRAPHRAG_ANSWERS, GROUND_TRUTHS)]

    print("\n=== PASO 3: BERTScore ===")
    baseline_bs = compute_bertscore(baseline_answers, GROUND_TRUTHS)
    graphrag_bs = compute_bertscore(GRAPHRAG_ANSWERS, GROUND_TRUTHS)

    print("\n=== PASO 4: RAGAS — Baseline ===")
    ragas_baseline = compute_ragas(QUESTIONS, baseline_answers, GROUND_TRUTHS)

    print("\n=== PASO 5: RAGAS — GraphRAG ===")
    ragas_graphrag = compute_ragas(QUESTIONS, GRAPHRAG_ANSWERS, GROUND_TRUTHS)

    # -----------------------------------------------------------------------
    # Resultados
    # -----------------------------------------------------------------------
    print_results_table(
        "BASELINE — LLM sin contexto (llama3.1:8b, conocimiento paramétrico)",
        QUESTION_IDS, QUESTIONS, baseline_answers,
        GROUND_TRUTHS, baseline_f1, baseline_rl, baseline_bs,
        ragas_baseline["answer_similarity"], ragas_baseline["answer_correctness"],
    )

    print_results_table(
        "GRAPHRAG — Pipeline completo (GraphRAG + llama3.1:8b)",
        QUESTION_IDS, QUESTIONS, GRAPHRAG_ANSWERS,
        GROUND_TRUTHS, graphrag_f1, graphrag_rl, graphrag_bs,
        ragas_graphrag["answer_similarity"], ragas_graphrag["answer_correctness"],
    )

    # Resumen delta
    def avg(lst): return sum(x for x in lst if x is not None) / max(len([x for x in lst if x is not None]), 1)
    print("=== RESUMEN COMPARATIVO ===")
    print(f"{'Métrica':<20} {'Baseline':>10} {'GraphRAG':>10} {'Delta':>10}")
    print("-" * 52)
    for name, bl, gr in [
        ("Token F1",       avg(baseline_f1), avg(graphrag_f1)),
        ("ROUGE-L",        avg(baseline_rl), avg(graphrag_rl)),
        ("BERTScore",      avg(baseline_bs), avg(graphrag_bs)),
        ("RAGAS Sim.",     avg(ragas_baseline["answer_similarity"]), avg(ragas_graphrag["answer_similarity"])),
        ("RAGAS Correct.", avg(ragas_baseline["answer_correctness"]), avg(ragas_graphrag["answer_correctness"])),
    ]:
        delta = gr - bl
        sign = "+" if delta >= 0 else ""
        print(f"{name:<20} {bl:>10.3f} {gr:>10.3f} {sign}{delta:>9.3f}")

    # Guardar resultados en JSON para usarlos en el LaTeX
    results_json = {
        "baseline": {
            "answers": baseline_answers,
            "token_f1": baseline_f1,
            "rouge_l": baseline_rl,
            "bertscore": baseline_bs,
            "ragas_similarity": ragas_baseline["answer_similarity"],
            "ragas_correctness": ragas_baseline["answer_correctness"],
        },
        "graphrag": {
            "answers": GRAPHRAG_ANSWERS,
            "token_f1": graphrag_f1,
            "rouge_l": graphrag_rl,
            "bertscore": graphrag_bs,
            "ragas_similarity": ragas_graphrag["answer_similarity"],
            "ragas_correctness": ragas_graphrag["answer_correctness"],
        },
    }
    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {out_path}")
