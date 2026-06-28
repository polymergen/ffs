# global_state_manager.py - Centralized thread-safe global state management
import threading
import copy
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager

class GlobalStateManager:
    """Centralized thread-safe manager for all global state variables"""
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Global state variables
        self._current_video_index = 0
        self._target_embedding = None
        self._clip_neg_prompt = ""
        self._clip_pos_prompt = ""
        self._frame_move = 0
        self._alpha = 1.0
        self._alpha2 = 1.0
        
        # Threading synchronization
        self._render_lock = threading.Lock()
        self._ui_update_lock = threading.Lock()
        
        # Event notifications
        self._state_change_callbacks: Dict[str, List[Callable]] = {}
    
    @contextmanager
    def synchronized(self):
        """Context manager for thread-safe access"""
        with self._lock:
            yield
    
    def register_callback(self, state_key: str, callback: Callable):
        """Register callback for state changes"""
        with self.synchronized():
            if state_key not in self._state_change_callbacks:
                self._state_change_callbacks[state_key] = []
            self._state_change_callbacks[state_key].append(callback)
    
    def _notify_callbacks(self, state_key: str, old_value: Any, new_value: Any):
        """Notify registered callbacks of state changes"""
        if state_key in self._state_change_callbacks:
            for callback in self._state_change_callbacks[state_key]:
                try:
                    callback(old_value, new_value)
                except Exception as e:
                    print(f"[GLOBAL_STATE:callback_error] Callback failed: {e}")
    
    # Current video index management
    def get_current_video_index(self) -> int:
        with self.synchronized():
            return self._current_video_index
    
    def set_current_video_index(self, index: int) -> bool:
        with self.synchronized():
            old_index = self._current_video_index
            self._current_video_index = index
            print(f"[GLOBAL_STATE:current_video] Changed from {old_index} to {index}")
            self._notify_callbacks('current_video_index', old_index, index)
            return True
    
    # Target embedding management
    def get_target_embedding(self):
        with self.synchronized():
            return copy.deepcopy(self._target_embedding) if self._target_embedding else None
    
    def set_target_embedding(self, embedding):
        with self.synchronized():
            old_embedding = self._target_embedding
            self._target_embedding = copy.deepcopy(embedding) if embedding else None
            print(f"[GLOBAL_STATE:target_embedding] Updated")
            self._notify_callbacks('target_embedding', old_embedding, self._target_embedding)
    
    # Frame movement control
    def get_frame_move(self) -> int:
        with self.synchronized():
            return self._frame_move
    
    def set_frame_move(self, move: int):
        with self.synchronized():
            old_move = self._frame_move
            self._frame_move = move
            print(f"[GLOBAL_STATE:frame_move] Changed from {old_move} to {move}")
            self._notify_callbacks('frame_move', old_move, move)
    
    # Alpha blending values
    def get_alpha(self) -> float:
        with self.synchronized():
            return self._alpha
    
    def set_alpha(self, alpha: float):
        with self.synchronized():
            old_alpha = self._alpha
            self._alpha = max(0.0, min(1.0, alpha))  # Clamp to [0,1]
            print(f"[GLOBAL_STATE:alpha] Changed from {old_alpha} to {self._alpha}")
            self._notify_callbacks('alpha', old_alpha, self._alpha)
    
    def get_alpha2(self) -> float:
        with self.synchronized():
            return self._alpha2
    
    def set_alpha2(self, alpha2: float):
        with self.synchronized():
            old_alpha2 = self._alpha2
            self._alpha2 = max(0.0, min(1.0, alpha2))  # Clamp to [0,1]
            print(f"[GLOBAL_STATE:alpha2] Changed from {old_alpha2} to {self._alpha2}")
            self._notify_callbacks('alpha2', old_alpha2, self._alpha2)
    
    # CLIP prompts management
    def get_clip_prompts(self) -> tuple:
        with self.synchronized():
            return self._clip_pos_prompt, self._clip_neg_prompt
    
    def set_clip_prompts(self, pos_prompt: str, neg_prompt: str):
        with self.synchronized():
            old_pos, old_neg = self._clip_pos_prompt, self._clip_neg_prompt
            self._clip_pos_prompt = pos_prompt
            self._clip_neg_prompt = neg_prompt
            print(f"[GLOBAL_STATE:clip_prompts] Updated prompts")
            self._notify_callbacks('clip_prompts', (old_pos, old_neg), (pos_prompt, neg_prompt))
    
    # Render lock management
    @contextmanager
    def render_lock(self):
        """Context manager for render operations"""
        print(f"[GLOBAL_STATE:render_lock] Acquiring render lock")
        with self._render_lock:
            print(f"[GLOBAL_STATE:render_lock] Render lock acquired")
            yield
        print(f"[GLOBAL_STATE:render_lock] Render lock released")
    
    # UI update lock management  
    @contextmanager
    def ui_update_lock(self):
        """Context manager for UI update operations"""
        with self._ui_update_lock:
            yield
    
    # Bulk state operations
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all current state"""
        with self.synchronized():
            return {
                'current_video_index': self._current_video_index,
                'target_embedding': copy.deepcopy(self._target_embedding),
                'clip_pos_prompt': self._clip_pos_prompt,
                'clip_neg_prompt': self._clip_neg_prompt,
                'frame_move': self._frame_move,
                'alpha': self._alpha,
                'alpha2': self._alpha2
            }
    
    def restore_state_snapshot(self, snapshot: Dict[str, Any]):
        """Restore state from snapshot"""
        with self.synchronized():
            print(f"[GLOBAL_STATE:restore] Restoring state snapshot")
            for key, value in snapshot.items():
                if key == 'current_video_index':
                    self.set_current_video_index(value)
                elif key == 'target_embedding':
                    self.set_target_embedding(value)
                elif key == 'clip_pos_prompt' and 'clip_neg_prompt' in snapshot:
                    self.set_clip_prompts(value, snapshot['clip_neg_prompt'])
                elif key == 'frame_move':
                    self.set_frame_move(value)
                elif key == 'alpha':
                    self.set_alpha(value)
                elif key == 'alpha2':
                    self.set_alpha2(value)

# Global instance
global_state = GlobalStateManager()

# Convenience functions for backward compatibility
def get_current_video_index() -> int:
    return global_state.get_current_video_index()

def set_current_video_index(index: int) -> bool:
    return global_state.set_current_video_index(index)

def get_target_embedding():
    return global_state.get_target_embedding()

def set_target_embedding(embedding):
    return global_state.set_target_embedding(embedding)

def get_frame_move() -> int:
    return global_state.get_frame_move()

def set_frame_move(move: int):
    return global_state.set_frame_move(move)

def get_alpha_values() -> tuple:
    return global_state.get_alpha(), global_state.get_alpha2()

def set_alpha_values(alpha: float, alpha2: float):
    global_state.set_alpha(alpha)
    global_state.set_alpha2(alpha2)
