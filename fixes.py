# Enhanced Fixes for race conditions and synchronization issues
import threading
import functools
from typing import Any, Callable, Dict, Optional

# Thread-safe global state lock
_global_state_lock = threading.RLock()

def synchronized(func):
    """Decorator for thread-safe global variable access"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _global_state_lock:
            return func(*args, **kwargs)
    return wrapper

def safe_video_access(operation: Callable, video_index: int = None, default=None) -> Any:
    """Safely access video data with comprehensive error handling"""
    global videos, current_video
    
    with _global_state_lock:
        try:
            if video_index is None:
                video_index = current_video
            
            # Bounds checking
            if len(videos) == 0:
                print(f"[SAFE_ACCESS:warning] No videos available")
                return default
            
            if video_index < 0 or video_index >= len(videos):
                print(f"[SAFE_ACCESS:error] Index {video_index} out of range [0, {len(videos)})")
                return default
            
            return operation(videos[video_index])
            
        except IndexError as e:
            print(f"[SAFE_ACCESS:error] IndexError: {e}, video_index={video_index}, videos_len={len(videos)}")
            return default
        except Exception as e:
            print(f"[SAFE_ACCESS:error] Unexpected error: {e}")
            return default

@synchronized
def safe_on_slider_move(value):
    """Thread-safely move slider with bounds checking"""
    global videos, current_video
    
    def update_frame_index(video):
        video["current_frame_index"] = int(value)
        print(f"[SLIDER:update] Frame index set to {value}")
    
    safe_video_access(update_frame_index)

@synchronized
def safe_edit_index(amount):
    """Thread-safely edit index with bounds checking"""
    global videos, current_video, slider
    
    def edit_operation(video):
        mini = 0
        maxi = video.get('frame_number', 0)
        new_index = video.get("current_frame_index", 0) + amount
        new_index = max(mini, min(new_index, maxi))
        video["current_frame_index"] = new_index
        print(f"[INDEX_EDIT:update] Frame index changed by {amount} to {new_index}")
        if 'slider' in globals():
            slider.set(new_index)
    
    safe_video_access(edit_operation)

@synchronized
def safe_run_it_please():
    """Thread-safely start rendering with bounds checking"""
    global count, videos, current_video, render_button, stop_rendering_button
    
    def start_rendering(video):
        print(f"[RENDER:start] Starting rendering for video {current_video}")
        video['rendering'] = True
        video['current_frame_index'] = 0
        video['count'] = -1
        
        # UI updates with error handling
        try:
            if 'render_button' in globals():
                render_button.config(state='disabled')
            if 'stop_rendering_button' in globals():
                stop_rendering_button.config(state='normal')
        except Exception as e:
            print(f"[RENDER:ui_error] UI update failed: {e}")
        
        if video.get('type') == 1 and 'cap' in video:
            try:
                video['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception as e:
                print(f"[RENDER:cap_error] Failed to reset video position: {e}")
    
    safe_video_access(start_rendering)

@synchronized
def safe_not_run_it_please():
    """Thread-safely stop rendering with bounds checking"""
    global count, videos, current_video, render_button, stop_rendering_button
    
    def stop_rendering(video):
        print(f"[RENDER:stop] Stopping rendering for video {current_video}")
        video['rendering'] = False
        video['current_frame_index'] = 0
        video['count'] = -1
        
        # UI updates with error handling
        try:
            if 'render_button' in globals():
                render_button.config(state='normal')
            if 'stop_rendering_button' in globals():
                stop_rendering_button.config(state='disabled')
        except Exception as e:
            print(f"[RENDER:ui_error] UI update failed: {e}")
        
        # Video writer cleanup
        if video.get('type') == 1:
            try:
                if 'cap' in video:
                    video['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                if 'out' in video:
                    video['out'].release()
                    
                # Recreate video writer if settings available
                if 'out_settings_for_resetting' in video:
                    settings = video['out_settings_for_resetting']
                    import cv2
                    video['out'] = cv2.VideoWriter(
                        settings['name_temp'], 
                        settings['fourcc'], 
                        settings['fps'], 
                        (settings['width'], settings['height']))
            except Exception as e:
                print(f"[RENDER:cleanup_error] Cleanup failed: {e}")
    
    safe_video_access(stop_rendering)

@synchronized
def safe_unselect_face():
    """Thread-safely unselect face with bounds checking"""
    global target_embedding, args, videos, current_video
    
    print(f"[FACE:unselect] Unselecting face")
    args['selective'] = ''
    target_embedding = None
    
    def reset_face_selection(video):
        video['old_number'] = -1
    
    safe_video_access(reset_face_selection)

@synchronized
def safe_main_loop_check(current_loop_video):
    """Thread-safely check if video should render in main loop"""
    global videos
    
    def get_rendering_status(video):
        return video.get('rendering', False)
    
    result = safe_video_access(get_rendering_status, current_loop_video, False)
    if result:
        print(f"[MAIN_LOOP:check] Video {current_loop_video} is rendering")
    return result

@synchronized
def safe_delete_current_video():
    """Thread-safely delete the current video"""
    global videos, current_video
    
    if len(videos) == 0:
        print(f"[DELETE:warning] No videos to delete")
        return False
    
    if current_video < 0 or current_video >= len(videos):
        print(f"[DELETE:error] Invalid current video index: {current_video}")
        return False
    
    print(f"[DELETE:execute] Deleting video at index {current_video}")
    
    # Clean up video resources before deletion
    try:
        video = videos[current_video]
        if 'cap' in video and video['cap'] is not None:
            video['cap'].release()
        if 'out' in video and video['out'] is not None:
            video['out'].release()
    except Exception as e:
        print(f"[DELETE:cleanup_error] Resource cleanup failed: {e}")
    
    # Remove the video
    videos.pop(current_video)
    
    # Adjust current_video index safely
    if len(videos) == 0:
        current_video = -1
        print(f"[DELETE:result] All videos deleted, current_video set to -1")
    elif current_video >= len(videos):
        current_video = len(videos) - 1
        print(f"[DELETE:result] Current video adjusted to {current_video}")
    
    return True

def safe_update_video_property(video_index: int, key: str, value: Any) -> bool:
    """Thread-safely update a video property"""
    def update_property(video):
        video[key] = value
        print(f"[UPDATE:property] Video {video_index}: {key} = {value}")
    
    return safe_video_access(update_property, video_index, False) is not None

def safe_get_video_property(video_index: int, key: str, default=None) -> Any:
    """Thread-safely get a video property"""
    def get_property(video):
        return video.get(key, default)
    
    return safe_video_access(get_property, video_index, default)
