import re
import os
import fitz
from thefuzz import fuzz

def clean_text(text):
    text = ' '.join(text.split())  # Normaliza espacios
    return text.strip()

def extract_links_with_detailed_debug(pdf_path):
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        links = page.get_links()
        
        current_text = None
        current_target_page = None
        current_details = None
        
        for link in links:
            if 'uri' in link:
                continue
            rect = link.get("from", None)
            text = page.get_textbox(rect).strip() if rect and page.get_textbox(rect) else "(No text)"
            text = clean_text(text)  # Normaliza espacios
            
            target_page = link.get("page", None)
            
            if current_text is None:
                current_text = text
                current_target_page = target_page
                current_details = link
            else:
                current_ends_with_number = re.search(r'\d+$', current_text.strip())
                new_starts_with_number = re.match(r'^\d+\.', text.strip())
                
                if current_ends_with_number or new_starts_with_number:
                    if current_text:
                        results.append({
                            "page_number": page_num + 1,
                            "text": current_text,
                            "target_page": current_target_page + 1 if current_target_page is not None else None,
                            "type": "Unrecognized or custom action",
                            "hyperlink": None,
                            "details": current_details,
                            "X": current_details['from'].x0,
                            "Nivel": 0,  # Inicializar nivel
                            "Id": ""    # Inicializar ID vacío
                        })
                    current_text = text
                    current_target_page = target_page
                    current_details = link
                else:
                    current_text += " " + text
                    current_details = link
        
        if current_text is not None:
            results.append({
                "page_number": page_num + 1,
                "text": current_text,
                "target_page": current_target_page + 1 if current_target_page is not None else None,
                "type": "Unrecognized or custom action",
                "hyperlink": None,
                "details": current_details,
                "X": current_details['from'].x0,
                "Nivel": 0,  # Inicializar nivel
                "Id": ""    # Inicializar ID vacío
            })
    
    return results

def organizar_indice(hyperlinks_data):
    if not hyperlinks_data:
        return hyperlinks_data
    
    def extract_number_structure(title):
        numbers = re.findall(r'\d+', title.split()[0])  # Extrae "1.2.3" → [1, 2, 3]
        return list(map(int, numbers)) if numbers else []
    
    # Inicializar el primer elemento
    hyperlinks_data[0]['Id'] = '1'
    hyperlinks_data[0]['Nivel'] = 0
    hyperlinks_data[0]['Padre'] = ""  # Asegurar que existe, incluso vacío
    current_hierarchy = [1]
    
    for i in range(1, len(hyperlinks_data)):
        prev = hyperlinks_data[i-1]
        curr = hyperlinks_data[i]
        curr['Padre'] = ""  # Inicializar siempre la clave (evita KeyError)
        
        prev_numbers = extract_number_structure(prev['text'])
        curr_numbers = extract_number_structure(curr['text'])
        
        if prev_numbers and curr_numbers:
            # Lógica basada en numeración (jerarquía clara)
            if len(curr_numbers) == len(prev_numbers):
                current_hierarchy = prev_numbers[:-1] + [prev_numbers[-1] + 1]
                curr['Nivel'] = len(curr_numbers) - 1
            elif len(curr_numbers) > len(prev_numbers):
                current_hierarchy = prev_numbers + [1]
                curr['Nivel'] = len(curr_numbers) - 1
            else:
                current_hierarchy = curr_numbers
                curr['Nivel'] = len(curr_numbers) - 1
            
            # Asignar padre si no es nivel 0
            if curr['Nivel'] > 0:
                parent_hierarchy = current_hierarchy[:-1]
                parent_id = '.'.join(map(str, parent_hierarchy))
                for item in hyperlinks_data[:i]:
                    if extract_number_structure(item['text']) == parent_hierarchy:
                        curr['Padre'] = f"El padre es: {item['Id']} {item['text']}"
                        break
        else:
            # Lógica basada en posición (X) si no hay numeración clara
            ancestor_idx = i - 1
            while ancestor_idx >= 0 and hyperlinks_data[ancestor_idx]['X'] > curr['X']:
                ancestor_idx -= 1
            
            if ancestor_idx < 0:
                current_hierarchy = [current_hierarchy[0] + 1]
                curr['Nivel'] = 0
            else:
                ancestor = hyperlinks_data[ancestor_idx]
                if curr['X'] > ancestor['X']:
                    current_hierarchy = list(map(int, ancestor['Id'].split('.'))) + [1]
                    curr['Nivel'] = ancestor['Nivel'] + 1
                    curr['Padre'] = f"El padre es: {ancestor['Id']} {ancestor['text']}"
                elif curr['X'] == ancestor['X']:
                    current_hierarchy = list(map(int, ancestor['Id'].split('.')))
                    current_hierarchy[-1] += 1
                    curr['Nivel'] = ancestor['Nivel']
                    if curr['Nivel'] > 0:
                        parent_idx = ancestor_idx
                        while parent_idx >= 0 and hyperlinks_data[parent_idx]['Nivel'] >= curr['Nivel']:
                            parent_idx -= 1
                        if parent_idx >= 0:
                            parent = hyperlinks_data[parent_idx]
                            curr['Padre'] = f"El padre es: {parent['Id']} {parent['text']}"
                else:
                    ancestor_hierarchy = list(map(int, ancestor['Id'].split('.')))
                    if len(ancestor_hierarchy) > 1:
                        current_hierarchy = ancestor_hierarchy[:-1]
                        current_hierarchy[-1] += 1
                        curr['Nivel'] = len(current_hierarchy) - 1
                        if curr['Nivel'] > 0:
                            parent_hierarchy = current_hierarchy[:-1]
                            parent_id = '.'.join(map(str, parent_hierarchy))
                            for item in hyperlinks_data[:i]:
                                if item['Id'] == parent_id:
                                    curr['Padre'] = f"El padre es: {item['Id']} {item['text']}"
                                    break
                    else:
                        current_hierarchy = [ancestor_hierarchy[0] + 1]
                        curr['Nivel'] = 0
        
        curr['Id'] = '.'.join(map(str, current_hierarchy))
    
    return hyperlinks_data

def limpiar_texto(texto):
    # Eliminar todos los números
    texto = re.sub(r'\d+', '', texto)
    # Eliminar puntos suspensivos (2 o más puntos)
    texto = re.sub(r'\.{2,}', '', texto)
    # Eliminar puntos solitarios (precedidos o seguidos por espacio o inicio/fin)
    texto = re.sub(r'(^| )\.( |$)', ' ', texto)
    # Eliminar posibles espacios múltiples resultantes
    texto = ' '.join(texto.split())
    # Eliminar espacios alrededor de guiones u otros caracteres si es necesario
    texto = re.sub(r'\s*-\s*', '-', texto)  # Opcional: para mantener "texto-texto"
    return texto.strip()

def normalize_title_text(title):
    """Normaliza el texto para comparación, eliminando puntuación y normalizando espacios"""
    # Convertir a minúsculas
    title = title.lower()
    # Eliminar caracteres especiales y puntuación (conservando letras, números y espacios)
    title = re.sub(r'[^\w\s]', '', title)
    # Normalizar espacios
    title = ' '.join(title.split())
    return title

def find_title_in_page(title, page_text, formatted_spans, fuzzy_threshold=85):
    """
    Encuentra un título en el texto verificando que esté en negrita con mejor manejo de formatos
    """
    # Normalizar el título de búsqueda
    clean_title = normalize_title_text(title)
    
    # Primero buscar coincidencia exacta en negrita con normalización
    for span in formatted_spans:
        if not span["is_bold"]:
            continue
            
        span_normalized = normalize_title_text(span["text"])
        if clean_title == span_normalized:
            pos = page_text.find(span["text"])
            if pos >= 0:
                return pos, len(span["text"])
    
    # Si no se encontró exacto, buscar fuzzy en negrita con normalización
    best_ratio = 0
    best_span = None
    
    for span in formatted_spans:
        if not span["is_bold"]:
            continue
            
        span_normalized = normalize_title_text(span["text"])
        ratio = fuzz.token_set_ratio(clean_title, span_normalized)
        
        if ratio > best_ratio and ratio >= fuzzy_threshold:
            best_ratio = ratio
            best_span = span
    
    if best_span:
        pos = page_text.find(best_span["text"])
        if pos >= 0:
            return pos, len(best_span["text"])
    
    # Último intento: buscar sin requerir negrita (pero con umbral más alto)
    if fuzzy_threshold > 70:
        for span in formatted_spans:
            span_normalized = normalize_title_text(span["text"])
            ratio = fuzz.token_set_ratio(clean_title, span_normalized)
            
            if ratio >= fuzzy_threshold + 10:  # Umbral más alto para texto no en negrita
                pos = page_text.find(span["text"])
                if pos >= 0:
                    return pos, len(span["text"])
    
    return -1, 0  # No se encontró coincidencia

def get_page_text_without_margins(doc, page_num):
    """Obtener el texto de una página excluyendo los márgenes y con información de formato"""
    if page_num >= len(doc):
        return "", []
    
    page = doc[page_num]
    
    # Ignorar páginas con más de 3 enlaces
    page_links = page.get_links()
    if len(page_links) > 3:
        return "", []
    
    # Obtener dimensiones y calcular márgenes
    page_width = page.rect.width
    page_height = page.rect.height
    margin_x = page_width * 0.1
    margin_y = page_height * 0.1
    
    # Crear rectángulo que excluye márgenes
    clip_rect = fitz.Rect(
        margin_x, margin_y, 
        page_width - margin_x, page_height - margin_y
    )
    
    # Extraer texto con información de formato
    try:
        blocks = page.get_text("dict", clip=clip_rect).get("blocks", [])
    except:
        return "", []
    
    formatted_text = []
    plain_text_parts = []
    
    for block in blocks:
        # Algunos bloques pueden no tener 'lines' (como imágenes)
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                formatted_text.append({
                    "text": span["text"],
                    "is_bold": "bold" in span["font"].lower(),
                    "rect": span["bbox"]
                })
                plain_text_parts.append(span["text"])
    
    # Crear texto plano uniendo todas las partes
    plain_text = "\n".join(plain_text_parts)
    return plain_text, formatted_text

def extract_text_between_sections_strict(pdf_path, links_data):
    doc = fitz.open(pdf_path)
    chunks = []
    
    # Preprocesar títulos: crear versiones alternativas para búsqueda
    for link in links_data:
        full_title = link['text']
        # Crear versión simplificada del título (última parte después de ":")
        simplified_title = full_title.split(':')[-1].strip()
        link['search_titles'] = [full_title, simplified_title]
    
    for i in range(len(links_data)):
        current_link = links_data[i]
        next_link = links_data[i+1] if i+1 < len(links_data) else None
        
        section_id = current_link['Id']
        start_page = current_link['target_page']
        
        # Si no hay página de destino, continuamos
        if start_page is None or start_page > len(doc):
            continue
            
        start_page_idx = start_page - 1
        start_page_text, formatted_spans = get_page_text_without_margins(doc, start_page_idx)
        if not start_page_text or not formatted_spans:
            continue
        
        # Buscar todos los posibles títulos de la sección actual
        title_positions = []
        for title_variant in current_link['search_titles']:
            title_idx, title_len = find_title_in_page(title_variant, start_page_text, formatted_spans)
            if title_idx >= 0:
                title_positions.append((title_idx, title_len))
        
        if not title_positions:
            continue
            
        # Tomar la primera ocurrencia del título
        title_idx, title_len = min(title_positions, key=lambda x: x[0])
        text_start = title_idx + title_len
        
        # Determinar fin del texto
        text_end = len(start_page_text)
        
        # Buscar inicio de la siguiente sección en la misma página
        if next_link and next_link['target_page'] == start_page:
            for next_title_variant in next_link['search_titles']:
                next_title_idx, _ = find_title_in_page(next_title_variant, start_page_text, formatted_spans)
                if next_title_idx > title_idx:
                    text_end = min(text_end, next_title_idx)
                    break
        
        # Extraer texto de la página actual
        section_text = start_page_text[text_start:text_end].strip()
        
        # Determinar la página final
        if next_link and next_link['target_page'] is not None:
            end_page_idx = next_link['target_page'] - 1  # Usamos la página anterior a donde empieza la siguiente sección
        else:
            # Para el último chunk, usar la última página del documento
            end_page_idx = len(doc) - 1
        
        # Asegurarnos de no ir más allá de la página de inicio
        end_page_idx = max(start_page_idx, end_page_idx)
        
        # Procesar páginas adicionales (si las hubiera)
        for page_idx in range(start_page_idx + 1, end_page_idx + 1):
            page_text, page_formatted_spans = get_page_text_without_margins(doc, page_idx)
            if page_text and page_formatted_spans:
                # Si estamos en la última página de esta sección y hay una siguiente sección, buscar su inicio
                if next_link and page_idx == end_page_idx:
                    for next_title_variant in next_link['search_titles']:
                        next_title_idx, _ = find_title_in_page(next_title_variant, page_text, page_formatted_spans)
                        if next_title_idx >= 0:
                            page_text = page_text[:next_title_idx].strip()
                            break
                
                if page_text:
                    section_text += "\n\n" + page_text
        
        if section_text:
            chunk = {
                "section_id": section_id,
                "section_title": current_link['text'],
                "cleaned_title": limpiar_texto(current_link['text']),
                "start_page": start_page,
                "end_page": end_page_idx + 1,
                "text": section_text,
                "level": current_link['Nivel'],
                "parent": current_link['Padre'],
                "title_found": True
            }
            chunks.append(chunk)
    
    return chunks

def save_chunks_to_file(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(f"======== SECCIÓN {chunk['section_id']}: {chunk['section_title']} ========\n")
            f.write(f"Nivel: {chunk['level']}\n")
            if chunk['parent']:
                f.write(f"Padre: {chunk['parent']}\n")
            f.write(f"Páginas: {chunk['start_page']} - {chunk['end_page']}\n")
            f.write(f"Título encontrado: {'Sí' if chunk['title_found'] else 'No'}\n\n")
            f.write(chunk['text'])
            f.write("\n\n" + "=" * 80 + "\n\n")

def process_pdf_strict(pdf_file_path, output_file):
    # Extraer los enlaces
    hyperlinks_data = extract_links_with_detailed_debug(pdf_file_path)
    
    # Organizar el índice y añadir relaciones jerárquicas
    hyperlinks_data = organizar_indice(hyperlinks_data)
    
    # Extraer el texto entre secciones de manera estricta
    chunks = extract_text_between_sections_strict(pdf_file_path, hyperlinks_data)
    
    
    # Guardar los chunks en un archivo
    save_chunks_to_file(chunks, output_file)
    
    return chunks, hyperlinks_data