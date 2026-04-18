from typing import Dict, List


class DecisionPolicy:
    def __init__(self, mode='all_to_server', max_active=None,
                 self_loop=False, bidirectional=True, custom_edges=''):
        self.mode = mode
        self.max_active = max_active
        self.self_loop = self_loop
        self.bidirectional = bidirectional
        self.custom_edges = custom_edges

    def _parse_custom_edges(self):
        edges = []
        if not self.custom_edges.strip():
            return edges
        items = self.custom_edges.split(',')
        for item in items:
            item = item.strip()
            if '->' not in item:
                continue
            s, r = item.split('->')
            edges.append((s.strip(), r.strip()))
        return edges

    def decide_edges(self, node_ids, server_id='server'):

        if self.mode == 'no_comm':
            return []  
        
        if self.mode == 'all_to_server':
            return [(nid, server_id) for nid in node_ids]

        if self.mode == 'fully_connected':
            edges = []
            for s in node_ids:
                for r in node_ids:
                    if not self.self_loop and s == r:
                        continue
                    edges.append((s, r))
            return edges

        if self.mode == 'ring':
            edges = []
            n = len(node_ids)
            for i, s in enumerate(node_ids):
                r = node_ids[(i + 1) % n]
                if not self.self_loop or s != r:
                    edges.append((s, r))
                if self.bidirectional:
                    edges.append((r, s))
            return list(dict.fromkeys(edges))

        if self.mode == 'chain':
            edges = []
            for i in range(len(node_ids) - 1):
                s, r = node_ids[i], node_ids[i + 1]
                edges.append((s, r))
                if self.bidirectional:
                    edges.append((r, s))
            return edges

        if self.mode == 'custom':
            return self._parse_custom_edges()

        return [(nid, server_id) for nid in node_ids]