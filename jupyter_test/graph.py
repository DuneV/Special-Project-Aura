import networkx as nx
from typing import List, Dict, Tuple, Any
import math
import time

# ontología

def build_ontology() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Conceptos del grafo generales
    concepts = [
        "Comida","Bebida","Ingrediente","Contenedor","Herramienta",
        "Fruta","Lácteo","Endulzante","Hielo"
    ]
    # Se definen como conceptos 
    for c in concepts:
        G.add_node(c, type="concept")

    # Jerarquía básica
    G.add_edge("Fruta","Ingrediente", key="subclass_of", relation="subclass_of")
    G.add_edge("Lácteo","Ingrediente", key="subclass_of", relation="subclass_of")
    G.add_edge("Endulzante","Ingrediente", key="subclass_of", relation="subclass_of")
    G.add_edge("Hielo","Ingrediente", key="subclass_of", relation="subclass_of")

    # Compatibilidades
    for k in ["Fruta","Lácteo","Endulzante","Hielo"]:
        G.add_edge(k,"Bebida", key="compatible_with", relation="compatible_with")
        G.add_edge("Bebida",k, key="compatible_with", relation="compatible_with")

    # Clases de YOLOV8
    classes = {
        "manzana": "Fruta",
        "banano": "Fruta",
        "limón": "Fruta",
        "naranja": "Fruta",
        "fresa": "Fruta",
        "leche": "Lácteo",
        "azúcar": "Endulzante",
        "miel": "Endulzante",
        "hielo": "Hielo",
        "vaso": "Contenedor",
        "jarra": "Contenedor",
        "licuadora": "Herramienta",
        "exprimidor": "Herramienta",
        "cuchillo": "Herramienta",
        "tabla": "Herramienta"
    }
    for cls, parent in classes.items():
        G.add_node(cls, type="class")
        G.add_edge(cls, parent, key="is_a", relation="is_a")

    # Afordancias
    # Contenedores → contener Bebida
    for container in ["vaso","jarra"]:
        G.add_edge(container, "Bebida", key="affords_contain", relation="affords", affordance="contener")

    # Herramientas → acciones
    G.add_node("mezclar", type="action")
    G.add_node("exprimir", type="action")
    G.add_node("cortar", type="action")
    G.add_node("servir", type="action")
    G.add_node("agregar", type="action")
    G.add_node("endulzar", type="action")
    G.add_node("enfriar", type="action")

    G.add_edge("licuadora","mezclar", key="affords", relation="affords")
    G.add_edge("exprimidor","exprimir", key="affords", relation="affords")
    G.add_edge("cuchillo","cortar", key="affords", relation="affords")

    # Ingredientes → posibles acciones
    for fruit in ["manzana","banano","limón","naranja","fresa"]:
        G.add_edge(fruit, "Bebida", key="can_be_used_to", relation="can_be_used_to", action="hacer_bebida")
    for s in ["azúcar","miel"]:
        G.add_edge(s, "endulzar", key="can_be_used_to", relation="can_be_used_to")
    G.add_edge("hielo","enfriar", key="can_be_used_to", relation="can_be_used_to")
    G.add_edge("leche","Bebida", key="can_be_used_to", relation="can_be_used_to", action="hacer_bebida")

    return G

def make_instance_id(label: str) -> str:
    return f"{label}#{int(time.time()*1000)%10_000_000}"

def update_scene(G: nx.MultiDiGraph,
                 detections: List[Dict[str, Any]],
                 world_frame: str = "world") -> None:
    """
    detections: [{"label": "vaso", "conf":0.86, "bbox":[x1,y1,x2,y2], "pose": (x,y,z)} ...]
    """
    if not G.has_node(world_frame):
        G.add_node(world_frame, type="frame")

    for det in detections:
        label = det["label"]
        if not G.has_node(label):  # si no estaba en la ontología, añádelo como class suelta
            G.add_node(label, type="class")

        inst = make_instance_id(label)
        G.add_node(inst, type="instance", class_of=label,
                   conf=det.get("conf",1.0),
                   bbox=det.get("bbox"),
                   pose=det.get("pose", (math.nan, math.nan, math.nan)),
                   last_seen=time.time())

        # instance -> class
        G.add_edge(inst, label, key="instance_of", relation="instance_of")
        # instance -> world pose
        G.add_edge(inst, world_frame, key="located_at", relation="located_at")

def intent_from_text(text: str) -> Dict[str, Any]:
    text = text.lower()
    # Simplisismo
    # API de chat
    if "refresco" in text or "bebida" in text or "jugo" in text:
        return {
            "goal_concept": "Bebida",
            "constraints": {"fría": True, "dulce": True},
            "preferred_ingredients": ["Fruta"],
            "required_container": "Contenedor"
        }
    return {"goal_concept": None}

def find_instances(G: nx.MultiDiGraph, class_name: str) -> List[str]:
    out = []
    for n, data in G.nodes(data=True):
        if data.get("type") == "instance":
            # sigue arista instance_of -> class
            for _, cls, k in G.out_edges(n, keys=True):
                if k == "instance_of" and cls == class_name:
                    out.append(n)
    return out

def any_instance_of_concept(G: nx.MultiDiGraph, concept: str, concept_to_classes_cache=None) -> List[Tuple[str,str]]:
    """
    Devuelve [(instancia, clase)] para clases que sean 'is_a/subclass_of' del concepto.
    """
    if concept_to_classes_cache is None:
        concept_to_classes_cache = {}

    if concept not in concept_to_classes_cache:
        # Todas las clases que apunten (is_a/subclass_of) hacia el concepto (o sus superclases)
        candidate_classes = set()
        for node, data in G.nodes(data=True):
            if data.get("type") == "class":
                # BFS hacia arriba para ver si llega al 'concept'
                seen = set([node])
                frontier = [node]
                reach = False
                while frontier:
                    u = frontier.pop()
                    for _, v, k, d in G.out_edges(u, keys=True, data=True):
                        if d.get("relation") in ("is_a","subclass_of") and v not in seen:
                            if v == concept:
                                reach = True
                            seen.add(v)
                            frontier.append(v)
                if reach:
                    candidate_classes.add(node)
        concept_to_classes_cache[concept] = candidate_classes

    results = []
    for cls in concept_to_classes_cache[concept]:
        for inst in find_instances(G, cls):
            results.append((inst, cls))
    return results

def plan_make_cold_drink(G: nx.MultiDiGraph, intent: Dict[str,Any]) -> Dict[str,Any]:
    if intent.get("goal_concept") != "Bebida":
        return {"feasible": False, "reason": "Intención no soportada"}
    
    containers = []
    for c in ["vaso","jarra"]:
        containers += find_instances(G, c)
    if not containers:
        return {"feasible": False, "reason": "No hay contenedor en escena (vaso/jarra)"}
    container = containers[0]

    fruits = any_instance_of_concept(G, "Fruta")
    dairy  = any_instance_of_concept(G, "Lácteo")
    sweet  = any_instance_of_concept(G, "Endulzante")
    ice    = any_instance_of_concept(G, "Hielo")

    ingredients = []
    if fruits: ingredients.append(fruits[0]) 
    elif dairy: ingredients.append(dairy[0])
    else:
        return {"feasible": False, "reason": "No hay ingredientes base (fruta o lácteo)"}

    if sweet:
        ingredients.append(sweet[0])
    if ice:
        ingredients.append(ice[0])

    tools = []
    if any(cls in ("limón","naranja") for _, cls in fruits):
        tools += find_instances(G, "exprimidor")
    tools += find_instances(G, "licuadora")

    steps = []
    if tools:
        tool = tools[0]
        steps.append({"action":"preparar_base", "using": tool, "with": [i for i,_ in ingredients]})
    else:
        knife = find_instances(G, "cuchillo")
        if knife and ingredients:
            steps.append({"action":"cortar", "using": knife[0], "with": [ingredients[0][0]]})
        steps.append({"action":"mezclar_manual", "with": [i for i,_ in ingredients]})

    steps.append({"action":"servir", "into": container})
    if any(cls == "Hielo" for _, cls in ingredients):
        steps.append({"action":"enfriar"})
    if any(cls == "Endulzante" for _, cls in ingredients):
        steps.append({"action":"endulzar"})

    return {
        "feasible": True,
        "container": container,
        "ingredients": ingredients,
        "steps": steps
    }

if __name__ == "__main__":
    G = build_ontology()

    # detecciones
    detections = [
        {"label":"vaso", "conf":0.92, "bbox":[100,100,160,200], "pose":(0.4, 0.2, 0.9)},
        {"label":"limón", "conf":0.88, "bbox":[200,120,240,170], "pose":(0.6, 0.2, 0.9)},
        {"label":"azúcar", "conf":0.80, "bbox":[260,110,300,160], "pose":(0.7, 0.2, 0.9)},
        {"label":"hielo", "conf":0.83, "bbox":[320,110,370,170], "pose":(0.8, 0.2, 0.9)},
        {"label":"exprimidor", "conf":0.77, "bbox":[380,110,430,190], "pose":(0.9, 0.2, 0.9)}
    ]
    update_scene(G, detections)

    intent = intent_from_text("hazme un refresco")
    plan = plan_make_cold_drink(G, intent)
    print(plan)
