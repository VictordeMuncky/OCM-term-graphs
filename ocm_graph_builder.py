import re
import networkx as nx

# Major subjects (from the HRAF OCM site) -> their section-head codes
MAJOR_SUBJECTS = {
    "001 Basic Information": ["100", "130", "140", "160", "360", "510"],
    "002 History, Prehistory And Culture Change": ["170", "180", "910"],
    "003 Language And Communication": ["190", "200", "210", "900"],
    "004 Economy, Food, And Resource Exploitation": ["220", "230", "240", "250", "260", "270", "310",
                                                     "420", "430", "440", "450", "460", "470"],
    "005 Technology And Material Culture": ["280", "290", "320", "330", "340", "350", "370", "380",
                                            "390", "400", "410", "480", "490", "500", "710"],
    "006 Marriage, Family, Kinship, And Social Organization": ["580", "590", "600", "610", "860"],
    "007 Social Relationships": ["550", "560", "570", "890", "920"],
    "008 Life Cycle": ["150", "850", "880"],
    "009 Sexuality And Reproduction": ["830", "840"],
    "010 Political Organization And Behavior": ["620", "630", "640", "650", "660"],
    "011 Justice, Law And Social Problems": ["670", "680", "690", "730"],
    "012 International And Interethnic Relations": ["700", "720"],
    "013 Religion": ["770", "780", "790"],
    "014 Health, Illness, Medicine And Death": ["740", "750", "760"],
    "015 Education And Knowledge": ["800", "810", "820", "870"],
    "016 Arts": ["300", "530"],
    "017 Recreation And Entertainment": ["520", "540"],
    "018 Information Sources And Research": ["110", "120"],
}

PARENT_W, RELATED_W = 1.0, 0.5

def section_head(code):
    """Section = first two digits; head is the xx0 code. Works for 4-digit codes too:
    5310 -> 530 (section 53 ran out of slots 531-539, so new codes became 5310, 5311)."""
    return code[:2] + "0"

def build_graph(file_name):
    wrdnet = nx.DiGraph()

    # Open and read the text file and add nodes get rid of related terms check, only add ocm terms from line[0] like in other file
    with open(file_name, 'r', encoding='utf-8') as file:
        for text_content in file.read().split("\n\n"):
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            if not lines:
                continue
            m = re.match(r"(\d{3,4})\s+(.+?)\s*$", lines[0])
            if not m:
                continue
            node_id, label = m.groups()
            wrdnet.add_node(node_id, label=f"{node_id} {label.strip()}")

    codes = set(wrdnet.nodes)

    head_to_major = {h: major for major, heads in MAJOR_SUBJECTS.items() for h in heads}
    for major in MAJOR_SUBJECTS:
        wrdnet.add_node(major, name=major, label=major, title=major, level="major", major=major)
    for code in codes:
        head = section_head(code)
        major = head_to_major.get(head, "Unassigned")
        wrdnet.nodes[code]["major"] = major
        if code == head:                                   # a section head like 530 ART
            wrdnet.nodes[code]["level"] = "section"
            if major != "Unassigned":
                wrdnet.add_edge(major, code, kind="parent", weight=PARENT_W)
        else:                                              # an ordinary code like 533 MUSIC
            wrdnet.nodes[code]["level"] = "code"
            if head in codes:
                wrdnet.add_edge(head, code, kind="parent", weight=PARENT_W)

    with open(file_name, 'r', encoding='utf-8') as file:
        for text_content in file.read().split("\n\n"):
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            if not lines:
                continue
            m = re.match(r"(\d{3,4})\s+(.+?)\s*$", lines[0])
            if not m:
                continue
            for line in lines:
                if line.startswith("Related Terms - "): line = line[len("Related Terms - "):]
                else: continue
                pairs = re.findall(r"(\d+)\s+([A-Z][A-Z\s]+)", line)
                for node, label in pairs:
                    if node in codes and node != m.group(1) and not wrdnet.has_edge(m.group(1), node):
                        wrdnet.add_edge(m.group(1), node, kind="related", weight=RELATED_W)

    return wrdnet

def related_only(G):
    related_graph = nx.DiGraph()

    for u, v, attrs in G.edges(data=True):
        if attrs["kind"] == "related":
            related_graph.add_edge(u, v)

    return related_graph

