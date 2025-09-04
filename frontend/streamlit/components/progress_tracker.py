"""
Progress Tracker for Real-time Analysis Updates
WBS 2.4: Professional progress indicators for 30+ second analyses
"""

# Standard library imports
import time
from datetime import datetime
from typing import Dict, Optional

# Third-party imports
import streamlit as st

class AnalysisProgressTracker:
    """Enhanced progress tracker with real-time updates and stages"""

    def __init__(self):
        self.stages = [
            ("validation", "Validating URL", 5),
            ("scraping", "Extracting content", 25),
            ("analysis", "AI analysis in progress", 60),
            ("reporting", "Generating insights", 90),
            ("completion", "Finalizing results", 100)
        ]
        self.current_stage = 0
        self.start_time = None
        self.estimated_duration = 45  # seconds

    def create_progress_interface(self) -> Dict:
        """Create enhanced progress UI components"""

        # Create containers for different UI elements
        header_container = st.container()
        progress_container = st.container()
        details_container = st.container()

        with header_container:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                status_placeholder = st.empty()
            with col2:
                time_placeholder = st.empty()
            with col3:
                cancel_placeholder = st.empty()

        with progress_container:
            # Main progress bar
            main_progress = st.progress(0)
            # Stage indicator
            stage_placeholder = st.empty()

        with details_container:
            # Expandable details section
            with st.expander("Progress Details", expanded=False):
                details_placeholder = st.empty()

        return {
            "status": status_placeholder,
            "time": time_placeholder,
            "cancel": cancel_placeholder,
            "main_progress": main_progress,
            "stage": stage_placeholder,
            "details": details_placeholder
        }

    def update_progress(self, ui_components: Dict, stage_index: int, substage_progress: float = 0):
        """Update progress with stage information"""

        if stage_index >= len(self.stages):
            stage_index = len(self.stages) - 1

        stage_id, stage_name, stage_end_percent = self.stages[stage_index]

        # Calculate overall progress
        if stage_index > 0:
            prev_percent = self.stages[stage_index - 1][2]
            progress_range = stage_end_percent - prev_percent
            overall_progress = prev_percent + (progress_range * substage_progress / 100)
        else:
            overall_progress = stage_end_percent * substage_progress / 100

        # Update UI components
        ui_components["main_progress"].progress(int(overall_progress))

        # Status update
        ui_components["status"].markdown(f"**{stage_name}**")

        # Time estimation
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if overall_progress > 5:  # Avoid division by very small numbers
                estimated_total = elapsed * 100 / overall_progress
                remaining = max(0, estimated_total - elapsed)
                ui_components["time"].metric(
                    "Time Remaining",
                    f"{int(remaining)}s"
                )

        # Stage indicator
        stage_indicators = []
        for i, (sid, sname, _) in enumerate(self.stages):
            if i < stage_index:
                stage_indicators.append(f"✓ {sname}")
            elif i == stage_index:
                stage_indicators.append(f"🔄 {sname}")
            else:
                stage_indicators.append(f"⏳ {sname}")

        ui_components["stage"].markdown("\n".join([f"**Stage {i+1}:** {indicator}"
                                                 for i, indicator in enumerate(stage_indicators)]))

        # Detailed progress
        progress_details = f"""
        **Current Stage:** {stage_name} ({overall_progress:.1f}% complete)
        **Elapsed Time:** {elapsed:.1f}s
        **Estimated Total:** {self.estimated_duration}s
        """
        ui_components["details"].markdown(progress_details)

    def start_tracking(self):
        """Start progress tracking"""
        self.start_time = datetime.now()
        self.current_stage = 0


class BackgroundTaskManager:
    """Manage background analysis tasks with progress updates"""

    def __init__(self):
        self.active_tasks = {}
        self.task_results = {}

    def start_analysis_task(self, task_id: str, url: str, config: Dict) -> None:
        """Start analysis task in background"""

        if task_id in self.active_tasks:
            return

        # Store task info in session state
        if 'background_tasks' not in st.session_state:
            st.session_state.background_tasks = {}

        st.session_state.background_tasks[task_id] = {
            'url': url,
            'config': config,
            'status': 'starting',
            'progress': 0,
            'stage': 'validation',
            'start_time': datetime.now(),
            'result': None,
            'error': None
        }

    def update_task_progress(self, task_id: str, stage: str, progress: int):
        """Update task progress"""
        if 'background_tasks' in st.session_state and task_id in st.session_state.background_tasks:
            st.session_state.background_tasks[task_id].update({
                'stage': stage,
                'progress': progress,
                'status': 'running'
            })

    def complete_task(self, task_id: str, result: any = None, error: str = None):
        """Mark task as completed"""
        if 'background_tasks' in st.session_state and task_id in st.session_state.background_tasks:
            st.session_state.background_tasks[task_id].update({
                'status': 'completed' if result else 'failed',
                'progress': 100,
                'result': result,
                'error': error,
                'end_time': datetime.now()
            })

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get current task status"""
        if 'background_tasks' in st.session_state:
            return st.session_state.background_tasks.get(task_id)
        return None

    def render_active_tasks(self):
        """Render active background tasks"""
        if 'background_tasks' not in st.session_state:
            return

        active_tasks = [
            (tid, task) for tid, task in st.session_state.background_tasks.items()
            if task['status'] in ['starting', 'running']
        ]

        if not active_tasks:
            return

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Background Tasks**")

        for task_id, task in active_tasks:
            with st.sidebar.expander(f"Analyzing {task['url'][:30]}...", expanded=True):
                st.progress(task['progress'])
                st.caption(f"Stage: {task['stage']}")

                elapsed = (datetime.now() - task['start_time']).total_seconds()
                st.caption(f"Running for {elapsed:.0f}s")

                if st.button(f"Cancel", key=f"cancel_{task_id}"):
                    st.session_state.background_tasks[task_id]['status'] = 'cancelled'
                    st.rerun()


def create_progress_simulator():
    """Create a progress simulator for testing"""

    def simulate_analysis_progress(ui_components, duration=30):
        """Simulate analysis progress over specified duration"""

        tracker = AnalysisProgressTracker()
        tracker.start_tracking()

        stages = tracker.stages
        stage_durations = [3, 8, 15, 4, 1]  # Duration for each stage in seconds

        for stage_idx, (stage_id, stage_name, stage_percent) in enumerate(stages):
            stage_duration = stage_durations[stage_idx]
            steps = 20  # Number of progress updates per stage

            for step in range(steps + 1):
                substage_progress = (step / steps) * 100
                tracker.update_progress(ui_components, stage_idx, substage_progress)

                time.sleep(stage_duration / steps)

                # Check if user cancelled
                if 'cancel_analysis' in st.session_state and st.session_state.cancel_analysis:
                    ui_components["status"].error("Analysis cancelled by user")
                    return False

        return True

    return simulate_analysis_progress
