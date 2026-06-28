# thread_safe_queue_manager.py - Thread-safe processing queue management
import threading
import queue
import time
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager

class ThreadSafeQueueManager:
    """Thread-safe manager for video processing queues"""
    
    def __init__(self):
        self._render_queue = queue.Queue()
        self._face_swap_queue = queue.Queue() 
        self._processing_queue = queue.Queue()
        
        # Thread management
        self._worker_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._queue_lock = threading.Lock()
        
        # Status tracking
        self._active_jobs: Dict[str, Any] = {}
        self._completed_jobs: Dict[str, Any] = {}
        self._failed_jobs: Dict[str, Any] = {}
        
        # Performance monitoring
        self._queue_stats = {
            'render_queue_size': 0,
            'face_swap_queue_size': 0,
            'processing_queue_size': 0,
            'jobs_completed': 0,
            'jobs_failed': 0
        }
    
    def start_workers(self, num_render_workers: int = 1, num_swap_workers: int = 2):
        """Start worker threads for processing queues"""
        print(f"[QUEUE_MGR:start] Starting {num_render_workers} render workers, {num_swap_workers} swap workers")
        
        # Start render workers
        for i in range(num_render_workers):
            worker = threading.Thread(target=self._render_worker, args=(i,), daemon=True)
            worker.start()
            self._worker_threads.append(worker)
        
        # Start face swap workers  
        for i in range(num_swap_workers):
            worker = threading.Thread(target=self._face_swap_worker, args=(i,), daemon=True)
            worker.start()
            self._worker_threads.append(worker)
    
    def shutdown(self):
        """Shutdown all worker threads gracefully"""
        print(f"[QUEUE_MGR:shutdown] Shutting down queue manager")
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5.0)
        
        print(f"[QUEUE_MGR:shutdown] All workers stopped")
    
    def add_render_job(self, video_index: int, callback: Optional[Callable] = None) -> str:
        """Add video to render queue"""
        job_id = f"render_{video_index}_{time.time()}"
        
        job_data = {
            'id': job_id,
            'video_index': video_index,
            'callback': callback,
            'timestamp': time.time(),
            'type': 'render'
        }
        
        with self._queue_lock:
            self._render_queue.put(job_data)
            self._queue_stats['render_queue_size'] = self._render_queue.qsize()
            print(f"[QUEUE_MGR:render] Added render job {job_id} for video {video_index}")
        
        return job_id
    
    def add_face_swap_job(self, video_index: int, frame_data: Dict, callback: Optional[Callable] = None) -> str:
        """Add frame to face swap queue"""
        job_id = f"swap_{video_index}_{time.time()}"
        
        job_data = {
            'id': job_id,
            'video_index': video_index, 
            'frame_data': frame_data,
            'callback': callback,
            'timestamp': time.time(),
            'type': 'face_swap'
        }
        
        with self._queue_lock:
            self._face_swap_queue.put(job_data)
            self._queue_stats['face_swap_queue_size'] = self._face_swap_queue.qsize()
            print(f"[QUEUE_MGR:swap] Added face swap job {job_id} for video {video_index}")
        
        return job_id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics"""
        with self._queue_lock:
            status = {
                'render_queue_size': self._render_queue.qsize(),
                'face_swap_queue_size': self._face_swap_queue.qsize(),
                'processing_queue_size': self._processing_queue.qsize(),
                'active_jobs': len(self._active_jobs),
                'completed_jobs': self._queue_stats['jobs_completed'],
                'failed_jobs': self._queue_stats['jobs_failed'],
                'worker_threads': len(self._worker_threads)
            }
        return status
    
    def _render_worker(self, worker_id: int):
        """Worker thread for processing render jobs"""
        print(f"[QUEUE_MGR:render_worker_{worker_id}] Started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get job with timeout
                job_data = self._render_queue.get(timeout=1.0)
                
                with self._queue_lock:
                    self._active_jobs[job_data['id']] = job_data
                    self._queue_stats['render_queue_size'] = self._render_queue.qsize()
                
                print(f"[QUEUE_MGR:render_worker_{worker_id}] Processing job {job_data['id']}")
                
                # Process the render job
                success = self._process_render_job(job_data)
                
                # Update job status
                with self._queue_lock:
                    if job_data['id'] in self._active_jobs:
                        del self._active_jobs[job_data['id']]
                    
                    if success:
                        self._completed_jobs[job_data['id']] = job_data
                        self._queue_stats['jobs_completed'] += 1
                        print(f"[QUEUE_MGR:render_worker_{worker_id}] Completed job {job_data['id']}")
                    else:
                        self._failed_jobs[job_data['id']] = job_data
                        self._queue_stats['jobs_failed'] += 1
                        print(f"[QUEUE_MGR:render_worker_{worker_id}] Failed job {job_data['id']}")
                
                # Call callback if provided
                if job_data.get('callback'):
                    try:
                        job_data['callback'](job_data, success)
                    except Exception as e:
                        print(f"[QUEUE_MGR:render_worker_{worker_id}] Callback error: {e}")
                
                self._render_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[QUEUE_MGR:render_worker_{worker_id}] Error: {e}")
        
        print(f"[QUEUE_MGR:render_worker_{worker_id}] Stopped")
    
    def _face_swap_worker(self, worker_id: int):
        """Worker thread for processing face swap jobs"""
        print(f"[QUEUE_MGR:swap_worker_{worker_id}] Started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get job with timeout
                job_data = self._face_swap_queue.get(timeout=1.0)
                
                with self._queue_lock:
                    self._active_jobs[job_data['id']] = job_data
                    self._queue_stats['face_swap_queue_size'] = self._face_swap_queue.qsize()
                
                print(f"[QUEUE_MGR:swap_worker_{worker_id}] Processing job {job_data['id']}")
                
                # Process the face swap job
                success = self._process_face_swap_job(job_data)
                
                # Update job status  
                with self._queue_lock:
                    if job_data['id'] in self._active_jobs:
                        del self._active_jobs[job_data['id']]
                    
                    if success:
                        self._completed_jobs[job_data['id']] = job_data
                        self._queue_stats['jobs_completed'] += 1
                        print(f"[QUEUE_MGR:swap_worker_{worker_id}] Completed job {job_data['id']}")
                    else:
                        self._failed_jobs[job_data['id']] = job_data
                        self._queue_stats['jobs_failed'] += 1
                        print(f"[QUEUE_MGR:swap_worker_{worker_id}] Failed job {job_data['id']}")
                
                # Call callback if provided
                if job_data.get('callback'):
                    try:
                        job_data['callback'](job_data, success)
                    except Exception as e:
                        print(f"[QUEUE_MGR:swap_worker_{worker_id}] Callback error: {e}")
                
                self._face_swap_queue.task_done()
                
            except queue.Empty:
                continue  
            except Exception as e:
                print(f"[QUEUE_MGR:swap_worker_{worker_id}] Error: {e}")
        
        print(f"[QUEUE_MGR:swap_worker_{worker_id}] Stopped")
    
    def _process_render_job(self, job_data: Dict) -> bool:
        """Process a render job (placeholder - implement actual logic)"""
        try:
            video_index = job_data['video_index']
            # TODO: Implement actual render logic here
            time.sleep(0.1)  # Simulate work
            return True
        except Exception as e:
            print(f"[QUEUE_MGR:render_job] Error processing render job: {e}")
            return False
    
    def _process_face_swap_job(self, job_data: Dict) -> bool:
        """Process a face swap job (placeholder - implement actual logic)"""
        try:
            video_index = job_data['video_index']
            frame_data = job_data['frame_data']
            # TODO: Implement actual face swap logic here
            time.sleep(0.05)  # Simulate work
            return True
        except Exception as e:
            print(f"[QUEUE_MGR:swap_job] Error processing face swap job: {e}")
            return False

# Global instance
queue_manager = ThreadSafeQueueManager()
