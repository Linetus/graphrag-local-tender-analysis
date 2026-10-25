import os
import re
from typing import List, Dict, Tuple, Optional

import openai
from dotenv import load_dotenv
# Cargar variables de entorno
load_dotenv()

# Configurar el cliente de OpenAI
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DocumentStrategyAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):  # Usar gpt-4.1-mini cuando esté disponible
        self.model_name = model_name

    def analyze_query(self, query: str, document_info: Dict) -> Dict:
        prompt = self._build_analysis_prompt(query, document_info)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de documentos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        content = response.choices[0].message.content
        return self._parse_decision(content)

    def _build_analysis_prompt(self, query: str, document_info: Dict) -> str:
        doc_count = document_info.get('doc_count', 1)
        avg_pages = document_info.get('avg_pages', 'unknown')
        doc_types = document_info.get('doc_types', ['PDF document'])

        return f"""
        As an expert document analysis agent, your task is to determine whether a user's query requires:

        1) PARTIAL DOCUMENT ACCESS (RAG approach): The query can be answered by retrieving and analyzing only specific relevant sections of the document(s).
           Examples: "What are the requirements for X?", "Tell me about section Y", "What's the price mentioned in the document?"

        2) FULL DOCUMENT ACCESS: The query requires understanding or analyzing the entire document(s) as a whole.
           Examples: "Summarize the entire document", "Compare all sections", "What's the overall structure?", "Find inconsistencies throughout the document"

        Information about the document(s):
        - Number of documents: {doc_count}
        - Average pages per document: {avg_pages}
        - Document types: {', '.join(doc_types)}

        User query: "{query}"

        Provide your decision in this exact format:
        DECISION: [PARTIAL/FULL]
        CONFIDENCE: [1-10]
        REASONING: [Your detailed reasoning]
        REQUIRES_GLOBAL_CONTEXT: [YES/NO]
        REQUIRES_SPECIFIC_SECTIONS: [YES/NO]
        """

    def _parse_decision(self, response: str) -> Dict:
        decision = {
            'strategy': 'unknown',
            'confidence': 0,
            'reasoning': '',
            'requires_global_context': False,
            'requires_specific_sections': False
        }

        match = re.search(r'DECISION:\s*(\w+)', response)
        if match:
            decision['strategy'] = 'full' if match.group(1).upper() == 'FULL' else 'partial'

        match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        if match:
            decision['confidence'] = int(match.group(1))

        match = re.search(r'REASONING:\s*(.*?)(?=\n\w+:|$)', response, re.DOTALL)
        if match:
            decision['reasoning'] = match.group(1).strip()

        match = re.search(r'REQUIRES_GLOBAL_CONTEXT:\s*(\w+)', response)
        if match:
            decision['requires_global_context'] = match.group(1).upper() == 'YES'

        match = re.search(r'REQUIRES_SPECIFIC_SECTIONS:\s*(\w+)', response)
        if match:
            decision['requires_specific_sections'] = match.group(1).upper() == 'YES'

        return decision

class SmartDocumentProcessor:
    def __init__(self, rag_system, model_name: str = "gpt-4o-mini"):  # Usar gpt-4.1-mini cuando esté disponible
        self.rag_system = rag_system
        self.model_name = model_name
        self.strategy_agent = DocumentStrategyAgent(model_name)

    def process_query(self, query: str, modo1: bool, visualize: bool = False) -> Dict:
        document_info = self._collect_document_info()
        decision = self.strategy_agent.analyze_query(query, document_info)

        print(f"\nQuery strategy decision: {decision['strategy'].upper()} (confidence: {decision['confidence']}/10)")
        print(f"Reasoning: {decision['reasoning']}")

        if decision['strategy'] == 'partial':
            if not modo1:
                print("Respuesta consolidada")
                response = self.rag_system.generate_consolidated_graph_response(query, visualize=visualize)
                return {'type': 'consolidated_rag', 'response': response}
            else:
                print("Respuesta individual")
                responses = self.rag_system.generate_graph_responses(query, visualize=visualize)
                return {'type': 'partial_rag', 'responses': responses}
        else:
            print("Using full document processing approach...")
            responses = self._process_full_documents(query)
            return {'type': 'full_document', 'responses': responses}

    def _collect_document_info(self) -> Dict:
        docs = self.rag_system.documents
        total_pages = 0
        doc_types = []

        for doc in docs:
            if hasattr(doc, 'chunks') and doc.chunks:
                max_page = max([chunk.get('end_page', 0) for chunk in doc.chunks])
                total_pages += max_page

            doc_type = doc.filename.split('.')[-1].upper() if '.' in doc.filename else 'Unknown'
            doc_types.append(doc_type)

        avg_pages = total_pages / len(docs) if docs else 0

        return {
            'doc_count': len(docs),
            'avg_pages': round(avg_pages, 1),
            'doc_types': doc_types
        }

    def _process_full_documents(self, query: str) -> Dict[str, str]:
        responses = {}

        for doc in self.rag_system.documents:
            full_content = self._get_full_document_content(doc)
            prompt = f"""
            Eres un asistente experto en analizar documentos. El usuario ha hecho la siguiente pregunta:

            Pregunta: {query}

            A continuación te proporciono el contenido completo del documento {doc.filename}. 
            Por favor, responde a la pregunta del usuario basándote en todo el documento:

            {full_content}

            Por favor, responde a la pregunta de manera clara y completa, basándote en todo el contenido del documento.
            """

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis de documentos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=6000
            )

            responses[doc.filename] = response.choices[0].message.content

        return responses

    def _get_full_document_content(self, doc) -> str:
        ordered_chunks = sorted(doc.chunks, key=lambda x: x['section_id'])
        sections = []
        for chunk in ordered_chunks:
            section_header = f"Sección {chunk['section_id']} ({chunk['start_page']}-{chunk['end_page']}): {chunk['cleaned_title']}"
            sections.append(f"{section_header}\n\n{chunk['text']}")
        return "\n\n".join(sections)

    def interactive_chat(self, visualize: bool = False):
        print(f"\nSmart Document Processing System (using {self.model_name}).")
        print(f"Loaded {len(self.rag_system.documents)} documents.")
        print("\nOpciones disponibles:")
        print("- Haz cualquier pregunta sobre los documentos")
        print("- El sistema decidirá automáticamente si usar RAG o procesamiento completo")
        print("- Escribe 'salir' para terminar\n")
        print("Para Respuestas RAG_ Modo 1 / Modo 2 , Respuestas individuales / Consolidadas , Modo 1 por defecto")
        viz_enabled = visualize
        modo1 = True

        while True:
            query = input("\nTu pregunta (o 'salir', 'visualizar si/no'): ").strip()

            if query.lower() in ['salir', 'exit', 'quit']:
                break

            if query.lower() == 'visualizar si':
                viz_enabled = True
                print("\nVisualización de grafos activada")
                continue

            if query.lower() == 'visualizar no':
                viz_enabled = False
                print("\nVisualización de grafos desactivada")
                continue

            if query.lower() == 'modo 1':
                modo1 = True
                print("\nRespuestas individuales activadas")
                continue

            if query.lower() == 'modo 2':
                modo1 = False
                print("\nRespuestas consolidadas activadas")
                continue

            if not query:
                continue

            print("\nAnalizando consulta y seleccionando estrategia óptima...")

            result = self.process_query(query, modo1, visualize=viz_enabled)

            print("\n" + "="*80 + "\n")

            if result['type'] == 'consolidated_rag':
                print("\nRESPUESTA PARCIAL (RAG):\n")
                print(result['response'])
            else:
                for filename, response in result['responses'].items():
                    print(f"\nDOCUMENTO: {filename}\n")
                    print(response)
                    print("\n" + "-"*60 + "\n")

            print("\nConsulta completada. Puedes hacer otra pregunta o escribir 'salir' para terminar.")