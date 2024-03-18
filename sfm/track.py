from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np

class Track:
    def __init__(self, frames=None, items=None):
        self.frames = frames if frames is not None else []
        self.items = items if items is not None else []
        self.frame_to_item = {frame_id: item for frame_id, item in zip(self.frames, self.items)}

    def append(self, frame_id, item):
        self.frames.append(frame_id)
        self.items.append(item)
        self.frame_to_item[frame_id] = item
    
    def has_frame(self, frame_id):
        return frame_id in self.frame_to_item

    def get_item(self, i):
        return self.items[i]

    def get_frame(self, i):
        return self.frames[i]
    
    def __getitem__(self, i):
        return self.get_item(i)

    def __len__(self):
        return len(self.items)
    
    def __str__(self):
        return "[" + ",".join([f"({f},{i})" for f,i in zip(self.frames, self.items)]) + "]"
    

def dfs(graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
        v: Tuple[int, int], used: Dict[Tuple[int, int], bool],
        tracks: List[Track]):
    used[v] = True
    tracks[-1].append(frame_id=v[0], item=v[1])

    for u in graph[v]:
        # check if we didn't use this frame and we didn't visit this vertex
        if not tracks[-1].has_frame(u[0]) and not used[u]:
            dfs(graph, u, used, tracks)

def get_tracks(
        frames_to_matches: Dict[Tuple[int, int], np.array],
        track_min_length=5) -> List[Track]:
    """
    Takes dictionary (frame_id1, frame_id2) -> list of matches
    and returns list of tracks
    """    
    graph = defaultdict(list)
    
    # Build graph
    for (frame_id1, frame_id2), matches in frames_to_matches.items():
        for match in matches:
            u = (frame_id1, match[0])
            v = (frame_id2, match[1])
            graph[u].append(v)
            graph[v].append(u)
    
    used = {v: False for v in graph}
    tracks = []
    for v in graph:
        if not used[v]:
            tracks.append(Track())
            dfs(graph, v, used, tracks)
    tracks = list(filter(lambda track: len(track) >= track_min_length, tracks))
    return tracks