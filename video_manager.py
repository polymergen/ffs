# video_manager.py - Video management class to replace global variables
import cv2
from typing import List, Optional, Dict, Any

class VideoManager:
    """Manages video list and current video selection without global variables"""
    
    def __init__(self):
        self.videos: List[Dict[str, Any]] = []
        self.current_video_index: int = -1
        self.target_embedding = None
        self.clip_neg_prompt = ""
        self.clip_pos_prompt = ""
        
    def add_video(self, video_data: Dict[str, Any]) -> int:
        """Add a video to the list and return its index"""
        self.videos.append(video_data)
        if self.current_video_index == -1:
            self.current_video_index = 0
        return len(self.videos) - 1
    
    def delete_current_video(self) -> bool:
        """Safely delete the current video from the list"""
        if len(self.videos) == 0 or not self.is_valid_index(self.current_video_index):
            return False
        
        # Remove the current video
        self.videos.pop(self.current_video_index)
        
        # Adjust current_video_index safely
        if len(self.videos) == 0:
            self.current_video_index = -1
        elif self.current_video_index >= len(self.videos):
            self.current_video_index = len(self.videos) - 1
        
        return True
    
    def get_current_video(self) -> Optional[Dict[str, Any]]:
        """Safely get the current video"""
        if self.is_valid_index(self.current_video_index):
            return self.videos[self.current_video_index]
        return None
    
    def set_current_video(self, index: int) -> bool:
        """Set current video by index"""
        if self.is_valid_index(index):
            self.current_video_index = index
            return True
        return False
    
    def is_valid_index(self, index: int) -> bool:
        """Check if index is valid for current video list"""
        return 0 <= index < len(self.videos)
    
    def get_video_count(self) -> int:
        """Get total number of videos"""
        return len(self.videos)
    
    def get_current_index(self) -> int:
        """Get current video index"""
        return self.current_video_index
    
    def get_video_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get video by index"""
        if self.is_valid_index(index):
            return self.videos[index]
        return None
    
    def update_current_video_property(self, key: str, value: Any) -> bool:
        """Update a property of the current video"""
        current = self.get_current_video()
        if current is not None:
            current[key] = value
            return True
        return False
    
    def get_current_video_property(self, key: str, default=None):
        """Get a property from the current video"""
        current = self.get_current_video()
        if current is not None:
            return current.get(key, default)
        return default
