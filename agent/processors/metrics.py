from loguru import logger
from pipecat.frames.frames import Frame, MetricsFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LatencyMonitor(FrameProcessor):
    """logs time-to-first-byte per pipeline stage; place last in the pipeline."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for metric in frame.data:
                if isinstance(metric, TTFBMetricsData):
                    logger.info(
                        "ttfb {} {:.0f}ms", metric.processor, metric.value * 1000
                    )
        await self.push_frame(frame, direction)
