#Fix respecto test 3, ahora la similitud se muestra en todos
import re
import os
from rank_bm25 import BM25Okapi
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama
import tiktoken  # Para contar tokens
from document_strategy_agent import SmartDocumentProcessor
from chunk_extractor import process_pdf_strict
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter

class NetworkXRAG:
    def __init__(self, pdf_document):
        """Inicializar el grafo basado en un documento PDF ya procesado"""
        self.pdf_document = pdf_document
        self.graph = nx.DiGraph()  # Grafo dirigido para representar jerarquía
        self.build_graph()

    def build_graph(self):
        """Construir el grafo a partir de los chunks del documento"""
        for chunk in self.pdf_document.chunks:
            section_id = chunk['section_id']
            self.graph.add_node(
                section_id, 
                title=chunk['cleaned_title'],
                text=chunk['text'],
                level=chunk['level'],
                page_range=(chunk['start_page'], chunk['end_page']),
                embedding_idx=self.pdf_document.chunks.index(chunk)  # Para acceder a embeddings después
            )
        for chunk in self.pdf_document.chunks: #Creamos las aristas y conectamos las relaciones familiares
            section_id = chunk['section_id']
            if '.' in section_id:
                parent_id = '.'.join(section_id.split('.')[:-1])
                if parent_id in self.graph:
                    self.graph.add_edge(parent_id, section_id, type='hierarchical', weight=1.0)

            parts = section_id.split('.')
            if len(parts) > 1: #Incrementamos para ir obteiendo las siguientes secciones
                next_number = str(int(parts[-1]) + 1)
                next_section = '.'.join(parts[:-1] + [next_number])
                if next_section in self.graph:
                    self.graph.add_edge(section_id, next_section, type='sequential', weight=0.8)

    def calculate_family_scores(self, query_subgraph: nx.DiGraph) -> Dict[str, float]:
        """Calcula scores combinados para cada nodo considerando familiares cercanos"""
        node_scores = {}
        
        for node in query_subgraph.nodes(): #Obtenemos los scores de los familiares y los ponderamos con los propios
            base_score = query_subgraph.nodes[node].get('relevance', 0.0)
            family_members = self._get_close_family_members(node) 
            family_max_score = 0.0
            if family_members:
                family_scores = [
                    query_subgraph.nodes[fm].get('relevance', 0.0) 
                    for fm in family_members 
                    if fm in query_subgraph
                ]
                family_max_score = max(family_scores) if family_scores else 0.0
            combined_score = (base_score * 0.5) + (family_max_score * 0.5)
            node_scores[node] = combined_score
        
        return node_scores


    def _get_close_family_members(self, node: str) -> List[str]:
            """Obtiene familiares cercanos (padre, hermanos, hijos directos), para hacer la propagación (y ponderación de scores)"""
            members = []
            if '.' in node: #Padre
                parent_id = '.'.join(node.split('.')[:-1])
                members.append(parent_id)

            parts = node.split('.') #Hermanos
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                siblings = [
                    n for n in self.graph.nodes() 
                    if n.startswith(parent + '.') and n != node
                ]
                members.extend(siblings)

            children = list(self.graph.successors(node)) #Hijos
            members.extend(children)
            
            return members

    def get_optimal_context(self, query_subgraph: nx.DiGraph, max_nodes: int = 20, min_threshold: float = 0.35) -> List[str]:
        """
        Obtiene Scores directos, aplica propagación familiar y combina con métricas de centralidad
        """
        real_direct_scores = { #Scores directos
            node: query_subgraph.nodes[node].get('relevance', 0.0)
            for node in query_subgraph.nodes()
        }
        family_scores = self.calculate_family_scores(query_subgraph) #Calcular scores familiares

        pagerank = nx.pagerank(query_subgraph, weight='weight') #Metricas de centralidad
        betweenness = nx.betweenness_centrality(query_subgraph, weight='weight')

        final_scores = {} #Puntuación final ponderada, se puede retocar para intentar mejorar el rendimiento
        for node in query_subgraph.nodes():
            direct_score = real_direct_scores[node]
            combined_score = family_scores[node]
            level = self.graph.nodes[node]['level']

            score = (
                combined_score * 0.6 + 
                pagerank[node] * 0.2 + 
                betweenness[node] * 0.2 +
                (1.0 / (level + 1)) * 0.3 
            )

            if direct_score > 0 or combined_score >= min_threshold:
                final_scores[node] = score

        selected_nodes = [] #Agrupamos los nodos
        processed_families = set()

        sorted_nodes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True) #Ordenamos segun la puntuación final

        for node, score in sorted_nodes:
            if len(selected_nodes) >= max_nodes:
                break

            family_root = node.split('.')[0]
            if family_root in processed_families:
                continue

            family_nodes = [ #Obtenemos los nodos de la familia
                n for n in query_subgraph.nodes()
                if n.startswith(family_root) and 
                (real_direct_scores[n] > 0 or family_scores[n] >= min_threshold)
            ]

            if len(selected_nodes) + len(family_nodes) <= max_nodes: #Intentamos dar prioridad a las familias completas
                selected_nodes.extend(family_nodes)
                processed_families.add(family_root)
            else:
                family_nodes_sorted = sorted(
                    family_nodes,
                    key=lambda x: final_scores.get(x, 0),
                    reverse=True
                )
                remaining_slots = max_nodes - len(selected_nodes)
                selected_nodes.extend(family_nodes_sorted[:remaining_slots])

        return selected_nodes[:max_nodes]

    def add_semantic_edges(self, threshold=0.65):
        """Añadir aristas basadas en similitud semántica entre secciones"""
        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            idx1 = self.graph.nodes[node1]['embedding_idx']
            embedding1 = self.pdf_document.content_embeddings[idx1].reshape(1, -1)
            
            for node2 in nodes[i+1:]:
                if nx.has_path(self.graph, node1, node2) or nx.has_path(self.graph, node2, node1): #Evitamos añadir aristas entre nodos ya conectados por jerarquía
                    continue
                    
                idx2 = self.graph.nodes[node2]['embedding_idx']
                embedding2 = self.pdf_document.content_embeddings[idx2].reshape(1, -1)

                sim = cosine_similarity(embedding1, embedding2)[0][0]

                if sim > threshold:
                    self.graph.add_edge(node1, node2, type='semantic', weight=sim)
                    self.graph.add_edge(node2, node1, type='semantic', weight=sim)  # Bidireccional
    

    def query_subgraph(self, query: str, top_k: int = 5, min_score: float = 0.35) -> nx.DiGraph:
        """Crear un subgrafo centrado en los nodos más relevantes para la consulta"""

        all_scored_chunks = self.pdf_document.query_chunks_hybrid(query, top_k_each=1000, min_similarity=0) #Obtenemos todos los chunks con scores

        central_chunks = [chunk for chunk in all_scored_chunks #Nodos centrales solo seran aquellos que superen el umbral
                        if chunk['similarity'] >= min_score or chunk['is_direct_match']]

        if len(central_chunks) < 3 and len(all_scored_chunks) > 0:
            central_chunks = sorted(all_scored_chunks, key=lambda x: x['similarity'], reverse=True)[:3]

        full_relevance_map = {chunk['section_id']: chunk['similarity'] for chunk in all_scored_chunks}
        full_source_map = {chunk['section_id']: chunk.get('match_source', 'unknown') for chunk in all_scored_chunks}

        central_nodes = [chunk['section_id'] for chunk in central_chunks]
        subgraph_nodes = set(central_nodes)

        for node in central_nodes: #Añadimos los ancestros
            # Añadir ancestros
            parts = node.split('.')
            for i in range(1, len(parts)):
                ancestor = '.'.join(parts[:i])
                if ancestor in self.graph:
                    subgraph_nodes.add(ancestor)

        for node in central_nodes: #Añadimos los descendientes
            for child in self.graph.successors(node):
                if self.graph.edges[node, child]['type'] == 'hierarchical':
                    subgraph_nodes.add(child)

        semantic_neighbors = set()
        for node in central_nodes:
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('type') == 'semantic' and edge_data.get('weight', 0) > 0.75:
                    semantic_neighbors.add(neighbor)
        
        subgraph_nodes.update(semantic_neighbors)

        query_subgraph = self.graph.subgraph(subgraph_nodes).copy() 

        for node_id in query_subgraph.nodes(): #Creamos el grafo y con scores
            if node_id in full_relevance_map:
                query_subgraph.nodes[node_id]['relevance'] = full_relevance_map[node_id]
                query_subgraph.nodes[node_id]['match_source'] = full_source_map[node_id]
                query_subgraph.nodes[node_id]['has_direct_score'] = True
            else:
                query_subgraph.nodes[node_id]['relevance'] = 0.0
                query_subgraph.nodes[node_id]['match_source'] = 'structural'
                query_subgraph.nodes[node_id]['has_direct_score'] = False
        
        return query_subgraph

    def get_context_with_hierarchy(self, selected_nodes: List[str]) -> List[Dict]:
        """
        Obtener el contexto completo para los nodos seleccionados, organizamos la información de manera jerárquica y filtrando por score
        """
        context_chunks = [] # Agrupar nodos por su raíz (primer número de sección)
        node_groups = {}
        for node in selected_nodes:
            root = node.split('.')[0]
            if root not in node_groups:
                node_groups[root] = []
            node_groups[root].append(node)

        for root, nodes in node_groups.items():
            nodes.sort() # Ordenar por ID de sección para mantener la jerarquía
            for node in nodes:
                idx = self.graph.nodes[node]['embedding_idx'] # Busca el chunk original
                chunk = self.pdf_document.chunks[idx].copy()
                if 'relevance' in self.graph.nodes[node]: # Añadir información de relevancia del grafo si existe 
                    chunk['graph_relevance'] = self.graph.nodes[node]['relevance']
                
                context_chunks.append(chunk)
        return context_chunks

    def visualize_query_graph(self, query_subgraph, filename='query_graph.png'):
            """Visualización que muestra scores combinados"""
            plt.figure(figsize=(16, 14))
            pos = nx.spring_layout(query_subgraph, seed=42, k=0.9, iterations=100)

            combined_scores = self.calculate_family_scores(query_subgraph) #Usamos los scores para la visualización
            colors = [] #Colores, formas etc.
            shapes = []
            color_map = {
                0: '#FF6B6B',  # Rojo claro para nivel 0
                1: '#4ECDC4',  # Turquesa para nivel 1
                2: '#45B7D1',  # Azul claro para nivel 2
                3: '#A593E0'   # Lila para nivel 3
            }
            shape_map = {
                0: 's',       # Cuadrado para nivel 0
                1: 'D',        # Diamante para nivel 1
                2: 'o',       # Círculo para nivel 2
                3: '^'        # Triángulo para nivel 3
            }
            for node in query_subgraph.nodes():
                level = self.graph.nodes[node]['level']
                colors.append(color_map.get(level, '#888888'))  # Gris por defecto
                shapes.append(shape_map.get(level, 'o'))

            node_sizes = []
            node_labels = {}
            for node in query_subgraph.nodes():
                combined_score = combined_scores.get(node, 0.3)
                node_sizes.append(1000 + 4000 * combined_score) #Nodo basado en score combinado 
                direct_score = query_subgraph.nodes[node].get('relevance', 'N/A')
                if isinstance(direct_score, float):
                    direct_score = f"{direct_score:.2f}"
                
                node_labels[node] = (
                    f"{node}\n"
                    f"Direct: {direct_score}\n"
                    f"Combined: {combined_score:.2f}"
                )

            edge_colors = [] #Dibujamos las aristas
            edge_styles = []
            edge_widths = []
            for u, v, data in query_subgraph.edges(data=True):
                if data['type'] == 'hierarchical':
                    edge_colors.append('#2F3C7E')  # Azul oscuro
                    edge_styles.append('solid')
                    edge_widths.append(2.0)
                elif data['type'] == 'sequential':
                    edge_colors.append('#6E6E6E')  # Gris
                    edge_styles.append('dashed')
                    edge_widths.append(1.0)
                elif data['type'] == 'semantic':
                    edge_colors.append('#FFA500')  # Naranja
                    edge_styles.append('dotted')
                    edge_widths.append(1.5 * data['weight'])  # Más grueso para mayor peso

            for i, (u, v) in enumerate(query_subgraph.edges()): #Dibujamos
                nx.draw_networkx_edges(
                    query_subgraph, pos, edgelist=[(u, v)],
                    edge_color=edge_colors[i], style=edge_styles[i],
                    width=edge_widths[i], alpha=0.8
                )

            for i, shape in enumerate(set(shapes)):
                nodes = [node for j, node in enumerate(query_subgraph.nodes()) if shapes[j] == shape]
                nx.draw_networkx_nodes(
                    query_subgraph, pos, nodelist=nodes,
                    node_color=[colors[j] for j, node in enumerate(query_subgraph.nodes()) if shapes[j] == shape],
                    node_size=[node_sizes[j] for j, node in enumerate(query_subgraph.nodes()) if shapes[j] == shape],
                    node_shape=shape,
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=0.8
                )

            nx.draw_networkx_labels( #Dibujamos etiquetas
                query_subgraph, pos, labels=node_labels,
                font_size=8, font_family='sans-serif',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.3')
            )
            legend_elements = [ #Leyenda
                plt.Line2D([0], [0], marker='s', color='w', label='Nivel 0 (Principal)',
                        markerfacecolor=color_map[0], markersize=12),
                plt.Line2D([0], [0], marker='D', color='w', label='Nivel 1',
                        markerfacecolor=color_map[1], markersize=12),
                plt.Line2D([0], [0], marker='o', color='w', label='Nivel 2',
                        markerfacecolor=color_map[2], markersize=12),
                plt.Line2D([0], [0], marker='^', color='w', label='Nivel 3+',
                        markerfacecolor=color_map[3], markersize=12),
                plt.Line2D([0], [0], color='#2F3C7E', linestyle='solid', label='Jerárquica', linewidth=2),
                plt.Line2D([0], [0], color='#6E6E6E', linestyle='dashed', label='Secuencial', linewidth=2),
                plt.Line2D([0], [0], color='#FFA500', linestyle='dotted', label='Semántica', linewidth=2)
            ]
            plt.legend(
                handles=legend_elements, 
                loc='upper right', 
                fontsize=10,
                title="Leyenda",
                title_fontsize=11,
                framealpha=0.9
            )
            plt.suptitle(
                "Grafo de Consulta con Scores Jerárquicos", 
                fontsize=16, 
                y=0.95,
                fontweight='bold'
            )
            plt.title(
                "Tamaño del nodo = Score combinado (50% propio + 50% mejor familiar)\n"
                "Color/forma = Nivel jerárquico | Tipo de línea = Relación",
                fontsize=10,
                pad=20
            )
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Grafo guardado como {filename}")

class PDFDocument:
    def __init__(self, pdf_path: str, model_name: str = "llama3.1:8b"):
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.model_name = model_name
        self.chunks = []
        self.title_embeddings = None
        self.content_embeddings = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 25000
        self.min_chunk_tokens = 0
        self._process_document()

    def _process_document(self):
        """Procesa el documento individual"""
        print(f"Procesando documento: {self.filename}")
        self.chunks, _ = process_pdf_strict(self.pdf_path, f"temp_{self.filename}_sections.txt")
        self._generate_embeddings()
        
    def _save_chunks_to_file(self, chunks: List[Dict], query: str, filename: str = "llm_chunks.txt"):
        """Guarda los chunks que se enviarán al LLM en un archivo"""
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n\n=== CONSULTA: {query} ===\n")
            f.write(f"Documento: {self.filename}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write("="*60 + "\n")
            
            for chunk in chunks:
                f.write(f"\nSección {chunk['section_id']} ({chunk['start_page']}-{chunk['end_page']}):\n")
                f.write(f"Título: {chunk['cleaned_title']}\n")
                f.write(f"Similitud: {chunk.get('similarity', 'N/A')}\n")
                f.write(f"Fuente coincidencia: {chunk.get('match_source', 'N/A')}\n")
                f.write(f"Coincidencia directa: {chunk.get('is_direct_match', 'N/A')}\n")
                f.write("\nContenido:\n")
                f.write(chunk['text'])
                f.write("\n" + "-"*50 + "\n")


    def _generate_embeddings(self):
        """Genera embeddings para el documento , distintos entre titulo y contenido"""
        if not self.chunks:
            raise ValueError("No hay chunks disponibles para generar embeddings")
        titles = [chunk['cleaned_title'] for chunk in self.chunks]
        self.title_embeddings = self.embedding_model.encode(titles)
        contents = [chunk['text'] for chunk in self.chunks]
        self.content_embeddings = self.embedding_model.encode(contents)

    def query_chunks_hybrid(self, query: str, top_k_each: int = 5, min_similarity: float = 0.45, max_combined: int = 10) -> List[Dict]:
        """Búsqueda híbrida con subchunking semántico, BM25, embeddings y reranking final con CrossEncoder"""
        if self.embedding_model is None:
            raise ValueError("Embedding model no disponible")
        query_embedding = self.embedding_model.encode([query])
        splitter = RecursiveCharacterTextSplitter( #Spliter para subchunking
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
        query_tokens = tokenize(query)

        enhanced_similarities_title = []
        enhanced_similarities_content = []
        top_subchunks_title = []
        top_subchunks_content = []

        for chunk in self.chunks:
            title_subs = splitter.split_text(chunk['cleaned_title'])
            content_subs = splitter.split_text(chunk['text'])

            if not title_subs: #Para titulos
                enhanced_similarities_title.append(0.0)
                top_subchunks_title.append([])
            else:
                title_embeddings = self.embedding_model.encode(title_subs)
                title_sims_emb = cosine_similarity(query_embedding, title_embeddings)[0]

                bm25_title = BM25Okapi([tokenize(t) for t in title_subs])
                title_sims_bm25 = bm25_title.get_scores(query_tokens)
                if max(title_sims_bm25) > 0:
                    title_sims_bm25 = [s / max(title_sims_bm25) for s in title_sims_bm25]

                combined = [
                    (title_subs[i], 0.675 * title_sims_emb[i] + 0.325 * title_sims_bm25[i])
                    for i in range(len(title_subs))
                ]
                combined.sort(key=lambda x: x[1], reverse=True)
                top_subchunks_title.append(combined[:3])
                scores = [score for _, score in combined[:3]]
                weights = [0.7, 0.2, 0.1]
                enhanced_similarities_title.append(sum(w * s for w, s in zip(weights, scores + [0]*(3-len(scores)))))

            if not content_subs: #Contenido
                enhanced_similarities_content.append(0.0)
                top_subchunks_content.append([])
            else:
                content_embeddings = self.embedding_model.encode(content_subs)
                content_sims_emb = cosine_similarity(query_embedding, content_embeddings)[0]

                bm25_content = BM25Okapi([tokenize(t) for t in content_subs])
                content_sims_bm25 = bm25_content.get_scores(query_tokens)
                if max(content_sims_bm25) > 0:
                    content_sims_bm25 = [s / max(content_sims_bm25) for s in content_sims_bm25]

                combined = [
                    (content_subs[i], 0.675 * content_sims_emb[i] + 0.325 * content_sims_bm25[i])
                    for i in range(len(content_subs))
                ]
                combined.sort(key=lambda x: x[1], reverse=True)
                top_subchunks_content.append(combined[:3])

                scores = [score for _, score in combined[:3]]
                weights = [0.7, 0.2, 0.1]
                enhanced_similarities_content.append(sum(w * s for w, s in zip(weights, scores + [0]*(3-len(scores)))))

        title_pairs = [(i, float(score), 'title') for i, score in enumerate(enhanced_similarities_title)]
        content_pairs = [(i, float(score), 'content') for i, score in enumerate(enhanced_similarities_content)]

        all_real_similarities = {}
        for idx, sim, src in title_pairs + content_pairs:
            if idx not in all_real_similarities or sim > all_real_similarities[idx][0]:
                all_real_similarities[idx] = (sim, src)

        all_chunks_with_scores = []
        for idx, (sim, src) in all_real_similarities.items():
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = sim
            chunk['match_source'] = src
            chunk['is_direct_match'] = True
            chunk['top_title_subchunks'] = top_subchunks_title[idx]
            chunk['top_content_subchunks'] = top_subchunks_content[idx]
            all_chunks_with_scores.append(chunk)

        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') #Reranker

        all_subchunks = []
        chunk_section_to_idx = {}
        for chunk_idx, chunk in enumerate(all_chunks_with_scores):
            chunk_section_to_idx[chunk['section_id']] = chunk_idx
            for text, score in chunk['top_title_subchunks'] + chunk['top_content_subchunks']:
                all_subchunks.append({
                    "text": text,
                    "original_score": score,
                    "chunk_id": chunk['section_id']
                })

        all_subchunks.sort(key=lambda x: x['original_score'], reverse=True)
        pairs = [(query, s['text']) for s in all_subchunks]

        if pairs: #Rerankeamos
            raw_scores = reranker.predict(pairs)
            min_score = float(min(raw_scores))
            max_score = float(max(raw_scores))
            if max_score == min_score:
                norm_scores = [1.0] * len(raw_scores)
            else:
                norm_scores = [(s - min_score) / (max_score - min_score) * (1.0 - min_similarity) + min_similarity for s in raw_scores]
            for i, s in enumerate(all_subchunks):
                s['reranked_score'] = norm_scores[i]

            reranked_map = {}
            for s in all_subchunks:
                reranked_map.setdefault(s['chunk_id'], []).append({
                    "text": s['text'],
                    "reranked_score": s['reranked_score']
                })

            for chunk in all_chunks_with_scores: #Actualizamos los scores
                chunk['reranked_top_subchunks'] = sorted(
                    reranked_map.get(chunk['section_id'], []),
                    key=lambda x: x['reranked_score'],
                    reverse=True
                )
                top_reranked = chunk['reranked_top_subchunks'][:3] #Ponderamos los scores (3 subchunks con mayor score)
                weights = [0.8, 0.15, 0.05]
                if len(top_reranked) == 1: #Si solo hay 1 le damos toda la score
                    scores = [top_reranked[0]['reranked_score']]
                    if scores[0] > 0.95:
                        chunk['similarity'] = scores[0]
                    else:
                        chunk['similarity'] = scores[0]
                else:
                    scores = [s['reranked_score'] for s in top_reranked]
                    padded_scores = scores + [0] * (3 - len(scores))
                    chunk['similarity'] = sum(w * s for w, s in zip(weights, padded_scores))

        filtered_chunks = [c for c in all_chunks_with_scores if c['similarity'] >= min_similarity]
        filtered_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        results = []
        processed_sections = set()
                
        for chunk in filtered_chunks:
            if chunk['section_id'] not in processed_sections and len(results) < max_combined:
                results.append({
                    'section_id': chunk['section_id'],
                    'cleaned_title': chunk['cleaned_title'],
                    'text': chunk['text'],
                    'start_page': chunk['start_page'],
                    'end_page': chunk['end_page'],
                    'level': chunk['level'],
                    'similarity': chunk['similarity'],  # Score directo
                    'match_source': chunk['match_source'],
                    'is_direct_match': chunk['is_direct_match'],
                    'reranked_top_subchunks': chunk.get('reranked_top_subchunks', [])
                })
                processed_sections.add(chunk['section_id'])

        with open("top3_subchunks_por_chunk.txt", "w", encoding="utf-8") as f: #Depuración, eliminar en version final
            for chunk in all_chunks_with_scores:
                f.write(f"\n--- Chunk ID: {chunk['section_id']} ---\n")
                f.write(f"Similarity score: {chunk['similarity']:.4f}\n")
                subs = chunk.get('reranked_top_subchunks') or []
                for i, sub in enumerate(subs[:3]):
                    f.write(f"{i+1}. Score: {sub['reranked_score']:.4f}\n")
                    f.write(f"   Text: {sub['text'][:400].strip()}...\n")
                f.write("\n")
        return results

class PDFDocumentWithNetworkX(PDFDocument):
    """Extensión de PDFDocument que integra capacidades de NetworkX"""
    
    def __init__(self, pdf_path: str, model_name: str = "llama3.1:8b"):
        super().__init__(pdf_path, model_name)
        self.graph_processor = NetworkXRAG(self)
        self.graph_processor.add_semantic_edges()
    
    def generate_graph_response(self,optimized_query: str, query: str, visualize: bool = False, model: Optional[str] = None, min_threshold: float = 0.35) -> str:
        """Genera respuesta usando el procesamiento del grafo mejorado con filtrado por umbral"""
        if model is None:
            model = self.model_name
        print(f"Consulta original: {query}\nConsulta optimizada: {optimized_query}")  # Para depuración eliminar en producción
        query_subgraph = self.graph_processor.query_subgraph(optimized_query, top_k=10, min_score=min_threshold) #Subgrafo centrado en la consulta optimizada

        selected_nodes = self.graph_processor.get_optimal_context(
            query_subgraph, 
            max_nodes=15, 
            min_threshold=min_threshold
    )
        optimized_chunks = self.graph_processor.get_context_with_hierarchy(selected_nodes)
        if visualize: #Visualizamos el grafo, lo mismo, depuración, se podria eliminar en producción o usar para mostrarlo
            selected_subgraph = query_subgraph.subgraph(selected_nodes)
            self.graph_processor.visualize_query_graph(
                selected_subgraph, 
                filename=f"graph_query_{self.filename}.png"
            )

        self._save_chunks_to_file(optimized_chunks, optimized_query, f"graph_llm_chunks_{self.filename}.txt")

        combined_scores = self.graph_processor.calculate_family_scores(query_subgraph)

        context = []
        level_chunks = {}
        
        for chunk in optimized_chunks:
            level = int(chunk.get('level', 0))
            if level not in level_chunks:
                level_chunks[level] = []
            level_chunks[level].append(chunk)

        for level in sorted(level_chunks.keys()):
            chunks = level_chunks[level]
            chunks.sort(key=lambda x: x['section_id'])
            
            for chunk in chunks:
                section_id = chunk['section_id']
                combined_score = combined_scores.get(section_id, 0.0)
                direct_score = chunk.get('graph_relevance', 0.0)

                if max(combined_score, direct_score) >= min_threshold: #Solo si uno de los scores supera el umbral
                    relevance_info = (
                        f" [Score directo: {direct_score:.2f} | "
                        f"Score combinado: {combined_score:.2f}]"
                    )
                    
                    context.append(
                        f"{'  ' * level}Sección {section_id}{relevance_info} "
                        f"({chunk['start_page']}-{chunk['end_page']}): {chunk['cleaned_title']}\n"
                        f"{chunk['text']}"
                    )
        
        context_text = "\n\n" + "\n\n".join(context)

        prompt = f"""
        Eres un asistente experto en analizar documentos. El usuario ha hecho la siguiente pregunta:
        
        Pregunta: {query}
        
        A continuación te proporciono información relevante extraída del documento {self.filename}, 
        organizada jerárquicamente según la estructura del documento. Se han incluido solo secciones 
        con score de relevancia de al menos {min_threshold}. Cada sección incluye:
        - Score directo: Relevancia calculada directamente para la consulta
        - Score combinado: Relevancia que considera también secciones relacionadas
        
        {context_text}
        
        Por favor, responde a la pregunta del usuario de manera clara y concisa, basándote en la información proporcionada. 
        Considera los scores de relevancia para ponderar la importancia de cada sección.
        Si la información no es suficiente para responder completamente, indícalo claramente.
        """
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        print("Tokens individual: ")
        print(len(encoding.encode(prompt)))
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'num_ctx': 10000
            }
        )
        return response['response']

class PDFRAGSystem:
    def __init__(self, pdf_paths: List[str] = None, model_name: str = "llama3.1:8b"):
        self.documents: List[PDFDocument] = []
        self.model_name = model_name
        
        if pdf_paths:
            for path in pdf_paths:
                try:
                    self.add_document(path)
                except Exception as e:
                    print(f"Error al procesar documento {path}: {str(e)}")

    def add_document(self, pdf_path: str):
        """Añade un nuevo documento al sistema"""
        doc = PDFDocument(pdf_path, self.model_name)
        self.documents.append(doc)
        print(f"Documento {doc.filename} añadido correctamente")
    
    def _save_consolidated_chunks_to_file(self, chunks: List[Dict], query: str, filename: str = "llm_consolidated_chunks.txt"):
        """Guarda los chunks consolidados que se enviarán al LLM en un archivo"""
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n\n=== CONSULTA CONSOLIDADA: {query} ===\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write("="*60 + "\n")
            
            for chunk in chunks:
                f.write(f"\nDocumento: {chunk['source_document']}\n")
                f.write(f"Sección {chunk['section_id']} ({chunk['start_page']}-{chunk['end_page']}):\n")
                f.write(f"Título: {chunk['cleaned_title']}\n")
                f.write(f"Score directo: {chunk.get('graph_relevance', 'N/A')}\n")
                f.write(f"Score combinado: {chunk.get('combined_score', 'N/A')}\n")
                f.write(f"Nivel jerárquico: {chunk.get('level', 'N/A')}\n")
                f.write("\nContenido:\n")
                f.write(chunk['text'])
                f.write("\n" + "-"*50 + "\n")

class PDFRAGSystemWithNetworkX(PDFRAGSystem):
    """Extensión de PDFRAGSystem para usar NetworkX"""
    
    def __init__(self, pdf_paths: List[str] = None, model_name: str = "llama3.1:8b"):
        self.documents: List[PDFDocumentWithNetworkX] = []
        self.model_name = model_name
        
        if pdf_paths:
            for path in pdf_paths:
                try:
                    self.add_document(path)
                except Exception as e:
                    print(f"Error al procesar documento {path}: {str(e)}")
    
    def add_document(self, pdf_path: str):
        """Añade un nuevo documento al sistema"""
        doc = PDFDocumentWithNetworkX(pdf_path, self.model_name)
        self.documents.append(doc)
        print(f"Documento {doc.filename} añadido correctamente con procesamiento de grafo")
    
    def generate_optimized_query(self, original_query: str, model: Optional[str] = None) -> str:
        """Agente que optimiza la consulta para GraphRAG eliminando elementos no esenciales"""
        if model is None:
            model = self.model_name
        
        prompt = f"""
        Eres un asistente especializado en optimizar consultas para sistemas de búsqueda en documentos. 
        Tu tarea es simplificar y enfocar la consulta del usuario manteniendo solo los términos clave 
        y conceptos esenciales para la búsqueda, eliminando:
        - Instrucciones sobre el estilo de respuesta (ej: "háblame extensamente", "explica detalladamente")
        - Palabras redundantes (ej: "todo lo que sepas sobre")
        - Elementos no relevantes para la búsqueda semántica
        -Especificaciones irrelevantes: En el proyecto, en el documento...
        - Si la pregunta contiene varios puntos, no hagas que se pierda información: ¿Qué perfiles técnicos o requisitos de equipo de trabajo se exigen al adjudicatario? → perfiles técnicos o requisitos de equipo de trabajo

        Ejemplos:
        - "Háblame de manera extensa sobre los requisitos funcionales de cada documento" → "Requisitos funcionales"
        - "Cuéntame todo lo que sepas de los requisitos funcionales y no funcionales del documento" → "Requisitos funcionales y no funcionales"
        - "Hablame de los precios en cada documento, además hazme un resumen combinado" → "Precios
        - "Hablame del alcance del documento / proyecto" → "Alcance"
        Por favor, procesa la siguiente consulta:
        
        Consulta original: {original_query}
        RESPONDE SOLO CON LA CONSULTA OPTIMIZADA, SIN EXPLICACIONES NI DETALLES ADICIONALES. UNICAMENTE LA STRING QUE DARIAMOS A RAG
        Consulta optimizada: """
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        print("Tokens Agente: ")
        print(len(encoding.encode(prompt)))
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'num_ctx': 2000
            }
        )
        
        return response['response'].strip()

    def generate_graph_responses(self, query: str, visualize: bool = False, model: Optional[str] = None) -> Dict[str, str]:
        optimized_query = self.generate_optimized_query(query, model)
        """Genera respuestas para todos los documentos usando análisis de grafos"""
        if not self.documents:
            return {"error": "No hay documentos cargados en el sistema"}
            
        if model is None:
            model = self.model_name 
        responses = {}
        
        for doc in self.documents:
            try:
                print(f"\nProcesando consulta en documento con grafo: {doc.filename}")
                response = doc.generate_graph_response(optimized_query ,query, visualize=visualize, model=model)
                responses[doc.filename] = response
            except Exception as e:
                responses[doc.filename] = f"Error al procesar documento: {str(e)}"
        
        return responses

    def generate_consolidated_graph_response(self, query: str, visualize: bool = False, model: Optional[str] = None, min_threshold: float = 0.35) -> str:
        """Genera una respuesta consolidada usando el análisis de grafos de todos los documentos"""
        if not self.documents:
            return "No hay documentos cargados en el sistema"
            
        if model is None:
            model = self.model_name
            
        optimized_query = self.generate_optimized_query(query, model) if self.documents else query
        print(f"Consulta original: {query}\nConsulta optimizada: {optimized_query}")  # Para depuración
        
        all_contexts = []
        subgraphs = []
        
        for doc in self.documents:
            try:
                print(f"\nProcesando consulta en documento: {doc.filename}")
                
                query_subgraph = doc.graph_processor.query_subgraph(optimized_query, top_k=20, min_score=min_threshold)
                subgraphs.append(query_subgraph)
                
                selected_nodes = doc.graph_processor.get_optimal_context(
                    query_subgraph, 
                    max_nodes=15, 
                    min_threshold=min_threshold
                )
                
                if visualize:
                    selected_subgraph = query_subgraph.subgraph(selected_nodes)
                    doc.graph_processor.visualize_query_graph(
                        selected_subgraph, 
                        filename=f"consolidated_graph_{doc.filename}.png"
                        
                    )
                context_chunks = doc.graph_processor.get_context_with_hierarchy(selected_nodes)
                combined_scores = doc.graph_processor.calculate_family_scores(query_subgraph)
                for chunk in context_chunks:
                    chunk.update({
                        'source_document': doc.filename,
                        'combined_score': combined_scores.get(chunk['section_id'], 0.0)
                    })
                
                all_contexts.extend(context_chunks)
            except Exception as e:
                print(f"Error al procesar documento {doc.filename}: {str(e)}")
        
        if not all_contexts:
            return "No se encontró información relevante en los documentos."
        
        level_chunks = {}
        for chunk in all_contexts:
            level = int(chunk.get('level', 0))
            if level not in level_chunks:
                level_chunks[level] = []
            level_chunks[level].append(chunk)
        
        context = []
        filtered_chunks = []  
        
        for level in sorted(level_chunks.keys()):
            chunks = level_chunks[level]
            chunks.sort(key=lambda x: x['section_id'])
            
            for chunk in chunks:
                direct_score = chunk.get('graph_relevance', 0.0)
                combined_score = chunk.get('combined_score', 0.0)
                
                if max(direct_score, combined_score) >= min_threshold:
                    filtered_chunks.append(chunk)  # Añadir a la lista de chunks filtrados
                    context.append(
                        f"Documento: {chunk['source_document']}\n"
                        f"{'  ' * level}Sección {chunk['section_id']} "
                        f"[Score directo: {direct_score:.2f} | Score combinado: {combined_score:.2f}] "
                        f"({chunk['start_page']}-{chunk['end_page']}): {chunk['cleaned_title']}\n"
                        f"{chunk['text']}"
                    )
        self._save_consolidated_chunks_to_file(filtered_chunks, query) # Depuración, guardamos los chunks consolidados
        
        if not context:
            return "No se encontró información relevante que supere el umbral de confianza."
        
        context_text = "\n\n".join(context)
        prompt = f"""
        Eres un asistente experto en analizar múltiples documentos. El usuario ha hecho la siguiente pregunta:
        Debes responder usando las siguientes pautas:
        1. Menciona explícitamente las fuentes (documentos) para cada pieza clave de información
        2. Señala cualquier contradicción o diferencia entre documentos
        3. Si la información es insuficiente, indícalo claramente
        4. Considera los scores de relevancia al ponderar la información
        
        A continuación te proporciono información relevante extraída de varios documentos,
        organizada jerárquicamente y filtrada por relevancia (umbral: {min_threshold}),:
        
        {context_text}
        
        Con esta información debes de responder a la siguiente pregunta del usuario de la mejor manera posible:
        Pregunta: {query}

        """
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        print("Tokens Conjunto: ")
        print(len(encoding.encode(prompt)))
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'num_ctx': 30000  # Aumentar contexto para múltiples documentos
            }
        )
        
        return response['response']

    def interactive_chat(self, model: Optional[str] = None, visualize: bool = False):
        """Modo interactivo de chat usando procesamiento de grafos"""
        if model is None:
            model = self.model_name
            
        print(f"\nSistema RAG con NetworkX para {len(self.documents)} documentos (usando {model}).")
        print("\nOpciones disponibles:")
        print("1. Respuestas individuales por documento usando grafos (predeterminado)")
        print("2. Respuesta consolidada de todos los documentos usando grafos")
        print("3. Visualizar grafos para depuración (crea imágenes PNG)")
        print("\nEscribe 'salir' para terminar, 'modo X' para cambiar el modo.\n")
        
        mode = 1  # Modo predeterminado: respuestas individuales
        visualize_graphs = visualize
        
        while True:
            query = input("\nTu pregunta (o 'salir', 'modo 1/2', 'visualizar si/no'): ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit']:
                break
                
            if query.lower() == 'modo 1':
                mode = 1
                print("\nModo cambiado: Respuestas individuales por documento")
                continue
                
            if query.lower() == 'modo 2':
                mode = 2
                print("\nModo cambiado: Respuesta consolidada de todos los documentos")
                continue

            if query.lower() == 'visualizar si':
                visualize_graphs = True
                print("\nVisualización de grafos activada")
                continue
                
            if query.lower() == 'visualizar no':
                visualize_graphs = False
                print("\nVisualización de grafos desactivada")
                continue
                
            if not query:
                continue
                
            print("\nBuscando información usando grafos...")
            
            if mode == 1: #Individuales
                responses = self.generate_graph_responses(query, visualize=visualize_graphs, model=model)
                
                print("\n" + "="*80 + "\n")
                for filename, response in responses.items():
                    print(f"\nDOCUMENTO: {filename}\n")
                    print(response)
                    print("\n" + "-"*60 + "\n")
            else: #Multiples documentos
                consolidated_response = self.generate_consolidated_graph_response(query, visualize=visualize_graphs, model=model)
                
                print("\n" + "="*80 + "\n")
                print("\nRESPUESTA CONSOLIDADA CON ANÁLISIS DE GRAFOS:\n")
                print(consolidated_response)
                print("\n" + "-"*60 + "\n")
            
            print("\nConsulta completada. Puedes hacer otra pregunta, cambiar el modo o escribir 'salir' para terminar.")

def main():
    base_path = "data/pliegos"  # run from the repository root; see data/README.md
    pdf_paths = [
        str(base_path + "/" +filename) 
        for filename in os.listdir(base_path) 
        if filename.lower().endswith('.pdf')
    ]

    print("Creando sistema RAG con NetworkX para múltiples documentos...")
    rag_system = PDFRAGSystemWithNetworkX(pdf_paths, model_name="llama3.1:8b")
    
    smart_processor = SmartDocumentProcessor(rag_system, model_name="llama3.1:8b")
    print("\nSistema listo con selección automática de estrategia.")
    smart_processor.interactive_chat(visualize=True)  # Activar visualización para depuración
    
if __name__ == "__main__":
    main()