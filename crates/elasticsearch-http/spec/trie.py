"""Python model of ES 8.19 / 9.5 PathTrie (org.elasticsearch.common.path.PathTrie)."""

MODES = ['EXPLICIT_NODES_ONLY', 'WILDCARD_ROOT_NODES_ALLOWED', 'WILDCARD_LEAF_NODES_ALLOWED', 'WILDCARD_NODES_ALLOWED']
W = '*'


def jsplit(path):
    """Java String.split("/"): trailing empty strings removed."""
    parts = path.split('/')
    while parts and parts[-1] == '':
        parts.pop()
    return parts


class Node:
    __slots__ = ('value', 'children')

    def __init__(self, value=None):
        self.value = value
        self.children = {}

    def insert(self, path, i, value):
        if i >= len(path):
            return
        tok = path[i]
        key = W if ('{' in tok and '}' in tok) or tok == '*' else tok
        node = self.children.get(key)
        if node is None:
            node = Node(value if i == len(path) - 1 else None)
            self.children[key] = node
        elif i == len(path) - 1:
            if node.value is None:
                node.value = value
            else:
                node.value = node.value + value
        node.insert(path, i + 1, value)

    def retrieve(self, path, i, mode):
        if i >= len(path):
            return None
        tok = path[i]
        node = self.children.get(tok)
        if node is None:
            if mode == 3:
                node = self.children.get(W)
                if node is None:
                    return None
                used_w = True
            elif mode == 1 and i == 1:
                node = self.children.get(W)
                if node is None:
                    return None
                used_w = True
            elif mode == 2 and i + 1 == len(path):
                node = self.children.get(W)
                if node is None:
                    return None
                used_w = True
            else:
                return None
        else:
            if i + 1 == len(path) and node.value is None and self.children.get(W) is not None and mode not in (0, 1):
                node = self.children[W]
                used_w = True
            elif i == 1 and node.value is None and self.children.get(W) is not None and mode == 1:
                node = self.children[W]
                used_w = True
            else:
                used_w = tok == W
        if i == len(path) - 1:
            return node.value
        v = node.retrieve(path, i + 1, mode)
        if v is None and not used_w and mode != 0:
            n2 = self.children.get(W)
            if n2 is not None:
                v = n2.retrieve(path, i + 1, mode)
        return v


class Trie:
    def __init__(self):
        self.root = Node()
        self.root_value = None

    def insert(self, path, value):
        parts = jsplit(path)
        if not parts:
            self.root_value = (self.root_value or []) + value
            return
        i = 1 if parts[0] == '' else 0
        self.root.insert(parts, i, value)

    def retrieve(self, path, mode):
        parts = jsplit(path)
        if not parts:
            return self.root_value
        i = 1 if parts[0] == '' else 0
        return self.root.retrieve(parts, i, mode)

    def retrieve_all(self, path):
        return [self.retrieve(path, m) for m in range(4)]
