import re
import os
import fitz
from thefuzz import fuzz

def clean_text(text):
    text = ' '.join(text.split())  # Normaliza espacios 
    return text.strip()

def get_page_margins(page):
    """Calcula los márgenes de la página (10% en cada lado)"""
    page_width = page.rect.width
    page_height = page.rect.height
    margin_x = page_width * 0.1
    margin_y = page_height * 0.1
    content_rect = fitz.Rect( #Rectangulo de contenido
        margin_x, margin_y, 
        page_width - margin_x, page_height - margin_y
    )
    
    return content_rect

def extract_links_with_detailed_debug(pdf_path):
    doc = fitz.open(pdf_path)
    results = []
    found_link_indices = False
    for page_num in range(len(doc)): #Busqueda con hipervinculos
        page = doc[page_num]
        links = page.get_links()
        content_rect = get_page_margins(page)
        current_text = None
        current_target_page = None
        current_details = None
        
        for link in links:
            if 'uri' in link:
                continue

            rect = link.get("from")
            link_rect = fitz.Rect(rect)
            if not (content_rect.intersects(link_rect)):
                continue  # Ignorar enlaces fuera de los márgenes

            found_link_indices = True
            text = page.get_textbox(rect).strip() if rect and page.get_textbox(rect) else ""
            text = clean_text(text)
            target_page = link.get("page")
            if current_text is None:
                # Ignorar entradas que sean solo números de página (sin título previo)
                if not re.match(r'^\d+\s*$', text):
                    current_text = text
                    current_target_page = target_page
                    current_details = link
            else:
                # Si el nuevo texto es solo un número de página, guardar la entrada actual y resetear
                if re.match(r'^\d+\s*$', text):
                    results.append({
                        "page_number": page_num + 1,
                        "text": current_text,
                        "target_page": current_target_page + 1 if current_target_page is not None else None,
                        "type": "link",
                        "details": current_details,
                        "X": current_details['from'].x0
                    })
                    current_text = None
                    current_target_page = None
                    current_details = None
                    continue

                current_ends_with_number = re.search(r'\d+$', current_text.strip())
                new_starts_with_number = re.match(r'^\d+\.', text.strip())

                if current_ends_with_number or new_starts_with_number:
                    if current_text:
                        results.append({
                            "page_number": page_num + 1,
                            "text": current_text,
                            "target_page": current_target_page + 1 if current_target_page is not None else None,
                            "type": "link",
                            "details": current_details,
                            "X": current_details['from'].x0
                        })
                    current_text = text
                    current_target_page = target_page
                    current_details = link
                else:
                    current_text += " " + text
                    current_details = link
        
        if current_text:
            results.append({
                "page_number": page_num + 1,
                "text": current_text,
                "target_page": current_target_page + 1 if current_target_page is not None else None,
                "type": "link",
                "details": current_details,
                "X": current_details['from'].x0
            })
    
    if found_link_indices:
        doc.close()
        return results

    results = []
    index_pattern = re.compile(
        r'^([\dA-Z]+\.?|\•|\*)\s*(.+?)\s*[\.\s]*\s*(\d+)\s*$',
        re.IGNORECASE
    )
    
    search_pages = min(6, len(doc))  # No más de 6 páginas o el total si es menor
    for page_num in range(search_pages):
        page = doc[page_num]
        content_rect = get_page_margins(page)
        page_text = page.get_text("text", clip=content_rect) #Sacar texto solo dentro del rectángulo
        for line in page_text.split('\n'):
            line = clean_text(line)
            match = index_pattern.search(line)
            
            if match:
                bullet = match.group(1) or ""
                text_content = match.group(2).strip()
                target_page = int(match.group(3))
                results.append({
                    "page_number": page_num + 1,
                    "text": f"{bullet} {text_content}".strip(),
                    "target_page": target_page,
                    "type": "text_index",
                    "details": None,
                })
    doc.close()
    return results

def extract_number_structure(title):
    """Extrae la estructura numérica del título (ej: "1.2.3 Título" -> [1, 2, 3])"""
    match = re.match(r'^((?:\d+\.)*\d+)', title.strip())
    if match: # Si encuentra una secuencia numérica con puntos (1.2.3), la separa
        numbering = match.group(1)
        return [int(num) for num in numbering.split('.')]
    
    match = re.match(r'^\s*(\d+)', title.strip()) # Busca simplemente el primer número (para casos como "1 - Título")
    if match:
        return [int(match.group(1))]
    
    return []

def organizar_indice(hyperlinks_data):
    if not hyperlinks_data:
        return hyperlinks_data

    # Inicializar primer elemento usando su estructura numérica real si la tiene
    first_numbers = extract_number_structure(hyperlinks_data[0]['text'])
    if first_numbers:
        hyperlinks_data[0]['Id'] = '.'.join(map(str, first_numbers))
        hyperlinks_data[0]['Nivel'] = len(first_numbers) - 1
        current_hierarchy = first_numbers
    else:
        hyperlinks_data[0]['Id'] = '1'
        hyperlinks_data[0]['Nivel'] = 0
        current_hierarchy = [1]
    hyperlinks_data[0]['Padre'] = ""
    
    for i in range(1, len(hyperlinks_data)):
        prev = hyperlinks_data[i-1]
        curr = hyperlinks_data[i]
        curr['Padre'] = ""  # Inicializar la clave
        prev_numbers = extract_number_structure(prev['text'])
        curr_numbers = extract_number_structure(curr['text'])
        if curr_numbers: # Usar directamente la estructura numérica del título

            current_hierarchy = curr_numbers
            curr['Nivel'] = len(curr_numbers) - 1 # Determinar el padre basado en la estructura numérica        
            if curr['Nivel'] > 0:
                parent_hierarchy = curr_numbers[:-1]
                parent_id = '.'.join(map(str, parent_hierarchy)) # Buscar el padre entre los elementos anteriores

                for idx, item in enumerate(hyperlinks_data[:i]):
                    item_numbers = extract_number_structure(item['text'])
                    if item_numbers and len(item_numbers) == len(parent_hierarchy): # Verificar si es el mismo prefijo numérico
                        if all(a == b for a, b in zip(item_numbers, parent_hierarchy)):
                            curr['Padre'] = f"El padre es: {item['Id']} {item['text']}"
                            break
        
        else:   # Caso 2: No hay estructura numérica explícita o no se pudo detectar correctamente
            if curr['type'] == 'text_index':
                if i > 0 and prev['type'] == 'text_index': # Elemento de índice textual - todos al mismo nivel a menos que tengan una clara estructura numérica
                    prev_id_parts = list(map(int, prev['Id'].split('.'))) # Si el anterior también es un índice textual, incrementamos el contador
                    current_hierarchy = prev_id_parts.copy()
                    current_hierarchy[-1] += 1
                else: # Si es el primer índice textual o el anterior es diferente, comienza nueva secuencia
                    current_hierarchy = [i + 1]  # Para evitar conflictos con IDs existentes
                
                curr['Nivel'] = 0  # Todos los índices textuales al nivel superior
            else: # Para hipervínculos reales, usamos la lógica basada en la posición X
                ancestor_idx = i - 1 
                while ancestor_idx >= 0 and hyperlinks_data[ancestor_idx]['X'] > curr['X']: # Buscar el elemento anterior con una posición X menor o igual
                    ancestor_idx -= 1
                
                if ancestor_idx < 0: # Si no hay ancestro válido, es un elemento de nivel superior
                    current_hierarchy = [hyperlinks_data[i-1]['Id'].split('.')[0]]
                    current_hierarchy = [int(current_hierarchy[0]) + 1]
                    curr['Nivel'] = 0
                else: # Hay un ancestro válido
                    ancestor = hyperlinks_data[ancestor_idx]
                    ancestor_id_parts = list(map(int, ancestor['Id'].split('.')))
                    
                    if curr['X'] > ancestor['X']: # Es un hijo del ancestro
                        current_hierarchy = ancestor_id_parts + [1]
                        curr['Nivel'] = ancestor['Nivel'] + 1
                        curr['Padre'] = f"El padre es: {ancestor['Id']} {ancestor['text']}"
                    elif curr['X'] == ancestor['X']: # Es un hermano del ancestro
                        current_hierarchy = ancestor_id_parts.copy()
                        current_hierarchy[-1] += 1
                        curr['Nivel'] = ancestor['Nivel']
                        
                        if curr['Nivel'] > 0: # Si no es de nivel 0, buscar su padre
                            parent_hierarchy = current_hierarchy[:-1]
                            parent_id = '.'.join(map(str, parent_hierarchy))
                            
                            for item in hyperlinks_data[:i]:
                                item_id_parts = item['Id'].split('.')
                                if len(item_id_parts) == len(parent_hierarchy) and all(int(a) == b for a, b in zip(item_id_parts, parent_hierarchy)):
                                    curr['Padre'] = f"El padre es: {item['Id']} {item['text']}"
                                    break

                    else: # Posición X menor: subir en la jerarquía  
                        estimated_level = 0 # Determinar el nivel basado en la indentación
                        for prev_item in reversed(hyperlinks_data[:i]):
                            if prev_item['X'] <= curr['X']:
                                if prev_item['X'] == curr['X']:
                                    # Mismo nivel
                                    estimated_level = prev_item['Nivel']
                                    current_hierarchy = list(map(int, prev_item['Id'].split('.')))
                                    current_hierarchy[-1] += 1
                                    break

                                elif prev_item['X'] < curr['X']:
                                    estimated_level = prev_item['Nivel'] + 1
                                    current_hierarchy = list(map(int, prev_item['Id'].split('.'))) + [1]
                                    curr['Padre'] = f"El padre es: {prev_item['Id']} {prev_item['text']}"
                                    break

                        if estimated_level == 0: # Si no se encontró un nivel adecuado, usar la lógica numérica
                            current_hierarchy = [int(hyperlinks_data[0]['Id'].split('.')[0]) + i]
                        
                        curr['Nivel'] = len(current_hierarchy) - 1
        
        curr['Id'] = '.'.join(map(str, current_hierarchy)) # Asignar el ID final
    
    return hyperlinks_data

def limpiar_texto(texto):
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\.{2,}', '', texto) # Eliminar puntos suspensivos (2 o más puntos)
    texto = re.sub(r'(^| )\.( |$)', ' ', texto)  # Eliminar puntos solitarios (precedidos o seguidos por espacio o inicio/fin)
    texto = ' '.join(texto.split()) # Eliminar posibles espacios múltiples resultantes
    texto = re.sub(r'\s*-\s*', '-', texto)
    return texto.strip()

def normalize_title_text(title):
    """Normaliza el texto para comparación, eliminando puntuación y normalizando espacios"""
    title = title.lower() # Convertir a minúsculas
    title = re.sub(r'[^\w\s]', '', title) # Eliminar caracteres especiales y puntuación (conservando letras, números y espacios)
    title = ' '.join(title.split()) # Normalizar espacios
    return title

def find_title_in_page(title, page_text, formatted_spans, fuzzy_threshold=85):
    """
    Encuentra un título en el texto verificando que esté en negrita con mejor manejo de formatos
    """
    clean_title = normalize_title_text(title)
    for span in formatted_spans: # Primero buscar coincidencia exacta en negrita con normalización
        if not span["is_bold"]:
            continue
            
        span_normalized = normalize_title_text(span["text"])
        if clean_title == span_normalized:
            pos = page_text.find(span["text"])
            if pos >= 0:
                return pos, len(span["text"])
    
    best_ratio = 0 # Si no se encontró exacto, buscar fuzzy en negrita con normalización
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
    
    if fuzzy_threshold > 70: # Último intento: buscar sin requerir negrita (pero con umbral más alto)
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
    
    page_links = page.get_links() # Ignorar páginas que parecen ser índices (muchos enlaces internos)
    internal_links = [l for l in page_links if 'uri' not in l]
    if len(internal_links) > 15:
        return "", []
    
    content_rect = get_page_margins(page) # Obtener rectángulo de contenido que excluye márgenes
    
    try: # Extraer texto con información de formato
        blocks = page.get_text("dict", clip=content_rect).get("blocks", [])
    except:
        return "", []
    
    formatted_text = []
    plain_text_parts = []
    
    for block in blocks:
        if "lines" not in block: # Algunos bloques pueden no tener 'lines' (como imágenes)
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                formatted_text.append({
                    "text": span["text"],
                    "is_bold": "bold" in span["font"].lower(),
                    "rect": span["bbox"]
                })
                plain_text_parts.append(span["text"])
    
    plain_text = "\n".join(plain_text_parts) # Crear texto plano uniendo todas las partes
    return plain_text, formatted_text

def extract_text_between_sections_strict(pdf_path, links_data):
    doc = fitz.open(pdf_path)
    chunks = []

    for link in links_data:
        full_title = link['text']
        simplified_title = full_title.split(':')[-1].strip() # Crear versión simplificada del título (última parte después de ":")
        link['search_titles'] = [full_title, simplified_title]
        print (simplified_title)

    for i in range(len(links_data)):
        current_link = links_data[i]
        next_link = links_data[i+1] if i+1 < len(links_data) else None
        
        section_id = current_link['Id']
        start_page = current_link['target_page']
        
        if start_page is None or start_page > len(doc): # Si no hay página de destino, continuamos
            continue
            
        start_page_idx = start_page - 1
        start_page_text, formatted_spans = get_page_text_without_margins(doc, start_page_idx)
        if not start_page_text or not formatted_spans:
            continue
        
        title_positions = [] # Buscar todos los posibles títulos de la sección actual
        for title_variant in current_link['search_titles']:
            title_idx, title_len = find_title_in_page(title_variant, start_page_text, formatted_spans)
            if title_idx >= 0:
                title_positions.append((title_idx, title_len))
        
        if not title_positions:
            continue
            
        title_idx, title_len = min(title_positions, key=lambda x: x[0]) # Tomar la primera ocurrencia del título
        text_start = title_idx + title_len
        text_end = len(start_page_text) # Determinar fin del texto
        
        if next_link and next_link['target_page'] == start_page: # Buscar inicio de la siguiente sección en la misma página
            for next_title_variant in next_link['search_titles']:
                next_title_idx, _ = find_title_in_page(next_title_variant, start_page_text, formatted_spans)
                if next_title_idx > title_idx:
                    text_end = min(text_end, next_title_idx)
                    break
        
        section_text = start_page_text[text_start:text_end].strip() # Extraer texto de la página actual
        
        if next_link and next_link['target_page'] is not None: # Determinar la página final
            end_page_idx = next_link['target_page'] - 1  # Usamos la página anterior a donde empieza la siguiente sección
        else:
            end_page_idx = len(doc) - 1 # Para el último chunk, usar la última página del documento

        end_page_idx = max(start_page_idx, end_page_idx) # Asegurarnos de no ir más allá de la página de inicio
        
        for page_idx in range(start_page_idx + 1, end_page_idx + 1): # Procesar páginas adicionales (en caso de haberlas)
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
    hyperlinks_data = extract_links_with_detailed_debug(pdf_file_path) # Extraer los enlaces
    hyperlinks_data = organizar_indice(hyperlinks_data) # Organizar el índice y añadir relaciones jerárquicas
    chunks = extract_text_between_sections_strict(pdf_file_path, hyperlinks_data) # Extraer el texto entre secciones de manera estricta
    save_chunks_to_file(chunks, output_file)  # Guardar los chunks en un archivo
    return chunks, hyperlinks_data