# thread_safe_video_manager.py - Thread-safe video management
import threading
import copy
from typing import List, Optional, Dict, Any, Callable
from contextlib import contextmanager

class ThreadSafeVideoManager:
    """Thread-safe video manager with proper synchronization"""
    
    def __init__(self):
        self._videos: List[Dict[str, Any]] = []
        self._current_video_index: int = -1
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._target_embedding = None
        self._clip_neg_prompt = ""
        self._clip_pos_prompt = ""
        
        # Event for notifying when videos list changes
        self._change_event = threading.Event()
        self._observers: List[Callable] = []
    
    @contextmanager
    def _synchronized(self):
        """Context manager for synchronized access"""
        with self._lock:
            yield
    
    def add_observer(self, callback: Callable) -> None:
        """Add observer for video list changes"""
        with self._synchronized():
            self._observers.append(callback)
    
    def _notify_observers(self) -> None:
        """Notify all observers of changes"""
        for observer in self._observers:
            try:
                observer()
            except Exception as e:
                print(f"[VIDEO_MGR:observer_error] Observer failed: {e}")
    
    def add_video(self, video_data: Dict[str, Any]) -> int:
        """Thread-safely add a video to the list"""
        with self._synchronized():
            print(f"[VIDEO_MGR:add] Adding video: {video_data.get('target_path', 'Unknown')}")
            self._videos.append(copy.deepcopy(video_data))
            if self._current_video_index == -1:
                self._current_video_index = 0
            index = len(self._videos) - 1
            self._notify_observers()
            return index
    
    def delete_current_video(self) -> bool:
        """Thread-safely delete the current video"""
        with self._synchronized():
            if len(self._videos) == 0 or not self._is_valid_index(self._current_video_index):
                print(f"[VIDEO_MGR:delete] Cannot delete - invalid index or empty list")
                return False
            
            print(f"[VIDEO_MGR:delete] Deleting video at index {self._current_video_index}")
            self._videos.pop(self._current_video_index)
            
            # Safely adjust current video index
            if len(self._videos) == 0:
                self._current_video_index = -1
            elif self._current_video_index >= len(self._videos):
                self._current_video_index = len(self._videos) - 1
            
            self._notify_observers()
            return True
    
    def delete_video_by_index(self, index: int) -> bool:
        """Thread-safely delete video by index"""
        with self._synchronized():
            if not self._is_valid_index(index):
                return False
            
            print(f"[VIDEO_MGR:delete] Deleting video at index {index}")
            self._videos.pop(index)
            
            # Adjust current index if necessary
            if index == self._current_video_index:
                if len(self._videos) == 0:
                    self._current_video_index = -1
                elif self._current_video_index >= len(self._videos):
                    self._current_video_index = len(self._videos) - 1
            elif index < self._current_video_index:
                self._current_video_index -= 1
                
            self._notify_observers()
            return True
    
    def get_current_video(self) -> Optional[Dict[str, Any]]:
        """Thread-safely get current video (deep copy to avoid race conditions)"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                return copy.deepcopy(self._videos[self._current_video_index])
            return None
    
    def get_current_video_reference(self) -> Optional[Dict[str, Any]]:
        """Get direct reference to current video (for performance critical operations)"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                return self._videos[self._current_video_index]
            return None
    
    def update_current_video_property(self, key: str, value: Any) -> bool:
        """Thread-safely update current video property"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                print(f"[VIDEO_MGR:update] Updating {key} = {value}")
                self._videos[self._current_video_index][key] = value
                return True
            return False
    
    def get_current_video_property(self, key: str, default=None) -> Any:
        """Thread-safely get current video property"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                return self._videos[self._current_video_index].get(key, default)
            return default
    
    def set_current_video_index(self, index: int) -> bool:
        """Thread-safely set current video index"""
        with self._synchronized():
            if self._is_valid_index(index):
                print(f"[VIDEO_MGR:select] Changing current video from {self._current_video_index} to {index}")
                self._current_video_index = index
                self._notify_observers()
                return True
            return False
    
    def get_current_video_index(self) -> int:
        """Thread-safely get current video index"""
        with self._synchronized():
            return self._current_video_index
    
    def get_video_count(self) -> int:
        """Thread-safely get video count"""
        with self._synchronized():
            return len(self._videos)
    
    def get_video_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Thread-safely get video by index (deep copy)"""
        with self._synchronized():
            if self._is_valid_index(index):
                return copy.deepcopy(self._videos[index])
            return None
    
    def get_all_videos(self) -> List[Dict[str, Any]]:
        """Thread-safely get all videos (deep copy)"""
        with self._synchronized():
            return copy.deepcopy(self._videos)
    
    def _is_valid_index(self, index: int) -> bool:
        """Check if index is valid"""
        return 0 <= index < len(self._videos)
    
    def is_current_video_rendering(self) -> bool:
        """Thread-safely check if current video is rendering"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                return self._videos[self._current_video_index].get('rendering', False)
            return False
    
    def safe_access_current_video(self, operation: Callable[[Dict[str, Any]], Any], default=None) -> Any:
        """Safely perform operation on current video with error handling"""
        with self._synchronized():
            try:
                if self._is_valid_index(self._current_video_index):
                    return operation(self._videos[self._current_video_index])
            except Exception as e:
                print(f"[VIDEO_MGR:safe_access_error] Operation failed: {e}")
            return default
    
    def batch_update_current_video(self, updates: Dict[str, Any]) -> bool:
        """Thread-safely update multiple properties at once"""
        with self._synchronized():
            if self._is_valid_index(self._current_video_index):
                print(f"[VIDEO_MGR:batch_update] Updating multiple properties: {list(updates.keys())}")
                self._videos[self._current_video_index].update(updates)
                return True
            return False

# Global instance
video_manager = ThreadSafeVideoManager()
