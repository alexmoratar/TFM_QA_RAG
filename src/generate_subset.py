import pandas as pd
import numpy as np

# Rutas de entrada y salida
base_path = "data"
queries_file = f"{base_path}/fiqa_queries.csv"
corpus_file = f"{base_path}/fiqa_corpus.csv"
qrels_file = f"{base_path}/fiqa_qrels.csv"

subset_queries_file = f"{base_path}/subset_queries.csv"
subset_corpus_file = f"{base_path}/subset_corpus.csv"
subset_qrels_file = f"{base_path}/subset_qrels.csv"

N_QUERIES = 300
MAX_DOCS = 3000
RANDOM_STATE = 42

df_queries = pd.read_csv(queries_file)
df_corpus = pd.read_csv(corpus_file)
df_qrels = pd.read_csv(qrels_file)

# Eliminar cabecera extra si la hay en qrels
if df_qrels.iloc[0].tolist() == ['query-id', 'corpus-id', 'score']:
    df_qrels = df_qrels.iloc[1:]

df_queries['_id'] = df_queries['_id'].astype(str)
df_corpus['_id'] = df_corpus['_id'].astype(str)
df_qrels['query_id'] = df_qrels['query_id'].astype(str)
df_qrels['doc_id'] = df_qrels['doc_id'].astype(str)

# Queries con al menos un qrel
queries_with_qrels = df_qrels['query_id'].unique()
valid_queries = df_queries[df_queries['_id'].isin(queries_with_qrels)]

# Muestreo de queries
subset_queries = valid_queries.sample(n=N_QUERIES, random_state=RANDOM_STATE)

# Docs relevantes para las queries seleccionadas
relevant_qrels = df_qrels[df_qrels['query_id'].isin(subset_queries['_id'])]
relevant_doc_ids = relevant_qrels['doc_id'].unique()
subset_docs = df_corpus[df_corpus['_id'].isin(relevant_doc_ids)]

# Si quieres añadir más documentos aleatorios hasta MAX_DOCS
if len(subset_docs) < MAX_DOCS:
    remaining = MAX_DOCS - len(subset_docs)
    other_docs = df_corpus[~df_corpus['_id'].isin(relevant_doc_ids)].sample(n=remaining, random_state=RANDOM_STATE)
    subset_docs = pd.concat([subset_docs, other_docs])

# Subset final de qrels
subset_qrels = relevant_qrels[
    relevant_qrels['doc_id'].isin(subset_docs['_id'])
]

# Guardar los archivos
subset_queries.to_csv(subset_queries_file, index=False)
subset_docs.to_csv(subset_corpus_file, index=False)
subset_qrels.to_csv(subset_qrels_file, index=False)

print(f"Subset generado: {len(subset_queries)} queries, {len(subset_docs)} docs, {len(subset_qrels)} qrels.")
