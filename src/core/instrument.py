from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
import time

from .substrate import CalibratedSubstrate

class InstrumentState(Enum):
    UNINITIALIZED = "uninitialized"
    CALIBRATING = "calibrating"
    READY = "ready"
    MEASURING = "measuring"
    ERROR = "error"

class AbstractInstrument(ABC):
    """
    Principle 2: Instruments as Scientific Apparatus, Not Tools.
    
    Each instrument follows a strict laboratory-grade protocol:
    Calibration -> Baseline -> Precision Verification -> Operation
    """
    
    def __init__(self, name: str):
        self.name = name
        self.state = InstrumentState.UNINITIALIZED
        self.calibration_data = {}
        self.baseline_readings = []
    
    def calibrate(self, ground_truth: CalibratedSubstrate) -> bool:
        """Step 1: Calibration against ground-truth substrates."""
        self.state = InstrumentState.CALIBRATING
        print(f"[{self.name}] Starting calibration...")
        success = self._perform_calibration(ground_truth)
        if success:
            self.state = InstrumentState.READY
            print(f"[{self.name}] Calibration successful.")
        else:
            self.state = InstrumentState.ERROR
            print(f"[{self.name}] Calibration failed.")
        return success

    def measure(self, substrate: CalibratedSubstrate) -> Dict[str, Any]:
        """Step 4: Operational Deployment."""
        if self.state != InstrumentState.READY:
            raise RuntimeError(f"Instrument {self.name} is not ready. Current state: {self.state}")
        
        self.state = InstrumentState.MEASURING
        reading = self._perform_measurement(substrate)
        self.state = InstrumentState.READY
        
        # Principle 3: Add introspection
        reading['meta'] = {
            'instrument': self.name,
            'timestamp': time.time(),
            'confidence': self._estimate_precision()
        }
        return reading

    @abstractmethod
    def _perform_calibration(self, ground_truth: CalibratedSubstrate) -> bool:
        pass

    @abstractmethod
    def _perform_measurement(self, substrate: CalibratedSubstrate) -> Dict[str, Any]:
        pass

    def _estimate_precision(self) -> float:
        """Self-reported precision estimate."""
        return 0.95  # Placeholder

class MetaInstrument(AbstractInstrument):
    """
    Principle 7: The Recursive Calibration Loop.
    
    An instrument that measures the measurement process itself.
    """
    
    def _perform_calibration(self, ground_truth: CalibratedSubstrate) -> bool:
        # Meta-instruments are axiomatically calibrated for this demo
        return True

    def _perform_measurement(self, substrate: CalibratedSubstrate) -> Dict[str, Any]:
        """
        In a real system, the 'substrate' here would be the runtime execution 
        of another instrument.
        """
        return {
            "type": "meta_measurement",
            "observation": "Nominal operation detected",
            "epistemic_status": "High confidence in measurement process"
        }
    
    def measure_measurement(self, target_instrument: AbstractInstrument, substrate: CalibratedSubstrate):
        """
        Measure how an instrument performs measurement.
        """
        print(f"[{self.name}] Observing {target_instrument.name}...")
        
        # 1. First-order measurement
        primary_reading = target_instrument.measure(substrate)
        
        # 2. Second-order measurement (Meta)
        process_observation = self.measure(substrate)
        
        return {
            "primary": primary_reading,
            "meta": process_observation
        }
