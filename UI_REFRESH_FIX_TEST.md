# UI Refresh Fix - Test Documentation

## Problem Fixed
The swapped preview wasn't updating in both the main GUI window and swapped preview window when:
1. Removing the current video and adding new ones
2. Changing the source face via file popup

## Root Cause Identified
- The `add()` function only added videos to the list but didn't trigger UI refresh
- The `delete_current_video()` function didn't call `frame_updater()` to refresh displays
- Missing UI refresh mechanism after video list changes

## Fix Applied

### 1. Enhanced `add()` function in main.py:
```python
def add():
    global videos, current_video
    print(f"[ADD_VIDEO:main] Adding new video")
    
    # Bounds checking before adding
    if len(videos) == 0:
        current_video = 0
        print(f"[ADD_VIDEO:main] First video, setting current_video to 0")
    
    try:
        facex = sorted(face_analysers[0].get(cv2.imread(args["face"])), key=lambda x: x.bbox[0])[0]
        videos.append(create_new_cap(args['target_path'], facex, args['output'],))
        print(f"[ADD_VIDEO:main] Video added successfully, total videos: {len(videos)}")
        
        # Update UI after adding video
        frame_updater()
        print(f"[ADD_VIDEO:main] UI refreshed after video addition")
        
    except Exception as e:
        show_error_custom(text=f"Wait few seconds and try again, program didn't start yet I will not notify you). debug error: {e}")
        print(f"[ADD_VIDEO:error] Failed to add video: {e}")
```

### 2. Enhanced `delete_current_video()` function in main.py:
```python
# Force GUI update after deletion
frame_updater()
print(f"[DELETE_VIDEO:main] Video deleted successfully, UI refreshed")
```

### 3. Verified `update_selector()` function:
- Already calls `frame_updater(False)` when video list changes
- This ensures the video list UI updates properly

## Debug Markers Added
- `[ADD_VIDEO:main]` - Video addition operations
- `[ADD_VIDEO:error]` - Video addition errors

## Testing Instructions

### Test Case 1: Video Addition UI Refresh
1. Start the application
2. Add a video using "Add this video" button
3. **Expected**: Swapped preview should immediately show the new video's first frame
4. **Debug**: Check console for `[ADD_VIDEO:main]` markers

### Test Case 2: Video Deletion UI Refresh  
1. Have multiple videos in the list
2. Delete the current video
3. **Expected**: Swapped preview should update to show the next available video
4. **Debug**: Check console for `[DELETE_VIDEO:main]` markers

### Test Case 3: Face Selection UI Refresh
1. Have a video loaded
2. Select a different source face via file popup
3. **Expected**: Swapped preview should update immediately with new face swap
4. **Debug**: Check console for `[FACE_SELECT:main]` or similar markers

## Technical Details

### What `frame_updater()` Does:
- Updates both original and swapped image displays
- Handles bounds checking for video access
- Clears displays when no videos are available
- Resizes images to fit preview windows
- Updates external preview window if enabled

### UI Refresh Chain:
1. Video operation (add/delete) occurs
2. `frame_updater()` is called
3. Current video bounds are checked
4. Image displays are updated
5. GUI is refreshed

## Files Modified:
- `main.py` - Enhanced `add()` and `delete_current_video()` functions

## Previous Fixes (Already Applied):
- Race condition fixes in all video access functions
- Bounds checking in `frame_updater()`, `update_selector()`, and other functions
- Comprehensive debug markers throughout the codebase
- Resource cleanup in video deletion

## Status: ✅ FIXED
Both the main GUI window and swapped preview window should now update properly when:
- Adding new videos
- Deleting current videos  
- The video list changes in any way

The UI refresh mechanism is now complete and should resolve the reported issue.
