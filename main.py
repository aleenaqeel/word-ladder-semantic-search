import streamlit as st
import numpy as np
import time
from collections import deque
import heapq
from typing import List, Tuple
import os
from functools import lru_cache

# ============================================================================ 
# WORD EMBEDDINGS
# ============================================================================

class WordEmbeddings:
    def __init__(self, filepath: str = "glove.100d.20000.txt"):
        self.word_to_index = {}
        self.index_to_word = []
        self.embeddings = None
        self.normalized_embeddings = None

        success = self.load_embeddings(filepath)
        if not success:
            st.error(f"❌ Could not load GloVe file: {filepath}")
            st.stop()

        self.normalize_embeddings()
        st.success(f"✅ Loaded {len(self.word_to_index)} words")

    def load_embeddings(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False

        embeddings_list = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    values = line.strip().split()
                    word = values[0]
                    vector = np.array(values[1:], dtype=np.float32)

                    self.word_to_index[word] = i
                    self.index_to_word.append(word)
                    embeddings_list.append(vector)

            self.embeddings = np.array(embeddings_list, dtype=np.float32)
            return True
        except:
            return False

    def normalize_embeddings(self):
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normalized_embeddings = self.embeddings / norms

    @lru_cache(maxsize=10000)
    def cosine_similarity(self, w1: str, w2: str) -> float:
        i1 = self.word_to_index.get(w1)
        i2 = self.word_to_index.get(w2)
        if i1 is None or i2 is None:
            return 0.0
        return float(np.dot(self.normalized_embeddings[i1],
                            self.normalized_embeddings[i2]))

    @lru_cache(maxsize=10000)
    def find_nearest_neighbors(self, word: str, k: int):
        idx = self.word_to_index.get(word)
        if idx is None:
            raise ValueError(f"Word '{word}' not found in embeddings.")

        word_vec = self.normalized_embeddings[idx]
        sims = self.normalized_embeddings @ word_vec
        sims[idx] = -1.0  # remove self
        top_indices = np.argpartition(-sims, k)[:k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        return tuple(self.index_to_word[i] for i in top_indices)

# ============================================================================ 
# SEARCH NODE
# ============================================================================

class SearchNode:
    __slots__ = ("word", "parent", "cost")

    def __init__(self, word, parent=None, cost=0):
        self.word = word
        self.parent = parent
        self.cost = cost

# ============================================================================ 
# SEARCH ALGORITHMS
# ============================================================================

class WordLadderSearch:
    def __init__(self, embeddings: WordEmbeddings, k_neighbors=20):
        self.embeddings = embeddings
        self.k = k_neighbors
        self.nodes_expanded = 0
        self.counter = 0  # tie-breaker for heap

    def reconstruct_path(self, node: SearchNode):
        path = []
        while node:
            path.append(node.word)
            node = node.parent
        return path[::-1]

    def heuristic(self, word, goal):
        return 1.0 - self.embeddings.cosine_similarity(word, goal)

    # ---------------- BFS ----------------
    def bfs(self, start, goal):
        start_time = time.time()
        self.nodes_expanded = 0

        visited = {start}
        queue = deque([SearchNode(start)])

        while queue:
            node = queue.popleft()
            self.nodes_expanded += 1

            if node.word == goal:
                path = self.reconstruct_path(node)
                return True, path, len(path)-1, self.nodes_expanded, time.time()-start_time

            neighbors = self.embeddings.find_nearest_neighbors(node.word, self.k)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(SearchNode(neighbor, node))

        return False, [], 0, self.nodes_expanded, time.time()-start_time

    # ---------------- DFS ----------------
    def dfs(self, start, goal, depth_limit=20):
        start_time = time.time()
        self.nodes_expanded = 0

        stack = [(SearchNode(start), 0)]
        visited = set()

        while stack:
            node, depth = stack.pop()
            if node.word in visited:
                continue

            visited.add(node.word)
            self.nodes_expanded += 1

            if node.word == goal:
                path = self.reconstruct_path(node)
                return True, path, len(path)-1, self.nodes_expanded, time.time()-start_time

            if depth < depth_limit:
                neighbors = self.embeddings.find_nearest_neighbors(node.word, self.k)
                for neighbor in neighbors:
                    stack.append((SearchNode(neighbor, node), depth+1))

        return False, [], 0, self.nodes_expanded, time.time()-start_time

    # ---------------- UCS ----------------
    def ucs(self, start, goal):
        start_time = time.time()
        self.nodes_expanded = 0
        self.counter = 0

        pq = []
        heapq.heappush(pq, (0, self.counter, SearchNode(start)))
        visited = {}

        while pq:
            cost, _, node = heapq.heappop(pq)
            if node.word in visited and visited[node.word] <= cost:
                continue

            visited[node.word] = cost
            self.nodes_expanded += 1

            if node.word == goal:
                path = self.reconstruct_path(node)
                return True, path, len(path)-1, self.nodes_expanded, time.time()-start_time

            neighbors = self.embeddings.find_nearest_neighbors(node.word, self.k)
            for neighbor in neighbors:
                self.counter += 1
                heapq.heappush(pq, (cost+1, self.counter, SearchNode(neighbor, node, cost+1)))

        return False, [], 0, self.nodes_expanded, time.time()-start_time

    # ---------------- Greedy ----------------
    def greedy(self, start, goal):
        start_time = time.time()
        self.nodes_expanded = 0
        self.counter = 0

        pq = []
        heapq.heappush(pq, (self.heuristic(start, goal), self.counter, SearchNode(start)))
        visited = set()

        while pq:
            _, _, node = heapq.heappop(pq)
            if node.word in visited:
                continue

            visited.add(node.word)
            self.nodes_expanded += 1

            if node.word == goal:
                path = self.reconstruct_path(node)
                return True, path, len(path)-1, self.nodes_expanded, time.time()-start_time

            neighbors = self.embeddings.find_nearest_neighbors(node.word, self.k)
            for neighbor in neighbors:
                self.counter += 1
                heapq.heappush(pq, (self.heuristic(neighbor, goal), self.counter, SearchNode(neighbor, node)))

        return False, [], 0, self.nodes_expanded, time.time()-start_time

    # ---------------- A* ----------------
    def astar(self, start, goal):
        start_time = time.time()
        self.nodes_expanded = 0
        self.counter = 0

        pq = []
        heapq.heappush(pq, (0, self.counter, SearchNode(start)))
        visited = {}

        while pq:
            f_score, _, node = heapq.heappop(pq)
            if node.word in visited and visited[node.word] <= node.cost:
                continue

            visited[node.word] = node.cost
            self.nodes_expanded += 1

            if node.word == goal:
                path = self.reconstruct_path(node)
                return True, path, len(path)-1, self.nodes_expanded, time.time()-start_time

            neighbors = self.embeddings.find_nearest_neighbors(node.word, self.k)
            for neighbor in neighbors:
                g = node.cost + 1
                h = self.heuristic(neighbor, goal)
                self.counter += 1
                heapq.heappush(pq, (g+h, self.counter, SearchNode(neighbor, node, g)))

        return False, [], 0, self.nodes_expanded, time.time()-start_time

# ============================================================================ 
# STREAMLIT GUI
# ============================================================================

def main():
    st.set_page_config(page_title="Word Ladder", page_icon="🔤")
    st.title("Word Ladder Search in Semantic Space")

    # Load embeddings once
    if 'embeddings' not in st.session_state:
        with st.spinner("Loading embeddings..."):
            st.session_state.embeddings = WordEmbeddings()

    start = st.text_input("Start word").lower().strip()
    goal = st.text_input("Goal word").lower().strip()
    algo = st.selectbox("Algorithm", ["BFS", "DFS", "UCS", "Greedy", "A*"])
    k = st.slider("k neighbors", 5, 50, 20)

    if st.button("Search"):
        # Check if start and goal words exist
        if start not in st.session_state.embeddings.word_to_index:
            st.error(f"❌ Start word '{start}' not found in embeddings.")
            return
        if goal not in st.session_state.embeddings.word_to_index:
            st.error(f"❌ Goal word '{goal}' not found in embeddings.")
            return

        searcher = WordLadderSearch(st.session_state.embeddings, k)

        with st.spinner("Searching..."):
            try:
                if algo == "BFS":
                    result = searcher.bfs(start, goal)
                elif algo == "DFS":
                    result = searcher.dfs(start, goal)
                elif algo == "UCS":
                    result = searcher.ucs(start, goal)
                elif algo == "Greedy":
                    result = searcher.greedy(start, goal)
                else:
                    result = searcher.astar(start, goal)
            except ValueError as e:
                st.error(str(e))
                return

        found, path, length, nodes, time_taken = result

        if found:
            st.success("✅ Path Found!")
            st.write(" → ".join(path))
        else:
            st.error("❌ No Path Found")

        st.metric("Steps", length)
        st.metric("Nodes Expanded", nodes)
        st.metric("Time", f"{time_taken:.4f}s")


if __name__ == "__main__":
    main()
