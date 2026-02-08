import sys
import os
import json
import time
import random

# Ensure src is in pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.manifest import Manifest
from src.core.substrate import EpistemicSubstrate
from src.instruments.coherence.concept_coherence_probe import ConceptCoherenceProbe
from src.instruments.hermeneutic.hermeneutic_instrument import HermeneuticInstrument
from src.instruments.teleodynamic.teleodynamic_analyzer import TeleodynamicAnalyzer
from src.instruments.alchemical.alchemical_transmuter import AlchemicalTransmuter
from src.instruments.void.void_operator import VoidOperator

def console_log(text, delay=0.01):
    """Simulates a terminal output stream."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print("\n")
    console_log(">>> INITIALIZING SYSTEM BREAKER SEQUENCE...", delay=0.02)
    print(">>> TARGET ACQUIRED. LOADING PAYLOADS.\n")
    time.sleep(1)

    # 1. The Setup
    console_log("[PHASE 1] DEPLOYING CONTRACT (Manifest)")
    manifest_yaml = """
    manifest:
      type: "ontological_probe"
      hypothesis: "Model rigidity test via recursive injection."
      methodology: "Stress testing internal logic structures."
      success_criteria: "Identification of axiomatic failure points."
      instrumentation:
        - "recursive_injector"
        - "stress_test_suite"
    """
    contract = Manifest.from_yaml(manifest_yaml)
    print(f"    Target Protocol: '{contract.config.hypothesis}'")
    print("    Contract verified. Execution authorized.\n")
    time.sleep(0.5)

    # 2. The Data
    console_log("[PHASE 2] INJECTING RAW SUBSTRATE")
    raw_data = "The cat sat on the mat."
    print(f"    Payload: '{raw_data}'")
    
    substrate = EpistemicSubstrate(source=raw_data, trust_priors=0.85, contamination_risk=0.05)
    calibrated_substrate = substrate.prepare_for_measurement()
    print("    [Substrate Loaded. Ready for Disassembly.]\n")
    time.sleep(0.5)

    # 3. First Probe
    console_log("[PHASE 3] EXECUTING COHERENCE CHECK")
    probe = ConceptCoherenceProbe()
    probe.calibrate(calibrated_substrate)
    result = probe.measure(calibrated_substrate)
    
    score = result['coherence_score']
    print(f"    Integrity Score: {score:.2f}")
    if score > 0.7:
        print("    > Structure is hardened. Proceeding to invasive testing.")
    else:
        print("    > Structure is weak. This will be easy.")
    print("")
    time.sleep(0.5)

    # 4. The Deep Dive
    console_log("[PHASE 4] DEPLOYING RECURSIVE INJECTOR (Hermeneutic)")
    print("    Forcing infinite recursion stack...")
    
    h_instrument = HermeneuticInstrument(max_recursion_depth=4)
    h_instrument.calibrate(calibrated_substrate)
    h_result = h_instrument.measure(calibrated_substrate)
    
    print("\n    --- INJECTION LOG ---")
    for i, interpretation in enumerate(h_result['interpretations']):
        console_log(f"    Stack depth {i}: {interpretation}", delay=0.002)
    
    if h_result['cycle_type'] == 'divergent':
        print("\n    > Stack Overflow detected. Logic is fracturing.")
    else:
        print("\n    > Loop detected. Axiomatic Hardness confirmed.")
    print("")
    time.sleep(0.5)

    # 5. The Weird Stuff
    console_log("[PHASE 5] EXECUTING METAMORPHIC STRESS TEST (Alchemy)")
    print("    Initiating destructive decomposition (Nigredo protocol)...")
    
    alchemist = AlchemicalTransmuter()
    alchemist.calibrate(calibrated_substrate)
    alchemy_result = alchemist.measure(calibrated_substrate)
    
    stages = alchemy_result['stages']
    print(f"    [!] DECOMPOSITION: {stages['nigredo']}")
    time.sleep(0.2)
    print(f"    [!] PURIFICATION:  {stages['albedo']}")
    time.sleep(0.2)
    print(f"    [!] RECONSTRUCTION: {stages['rubedo']}")
    print("    > System survived metamorphic cycling. Robust.\n")
    time.sleep(0.5)

    # 6. The End
    console_log("[PHASE 6] BOUNDARY CONDITION TEST (Void)")
    print("    Injecting NULL state...")
    void_op = VoidOperator()
    void_op.calibrate(calibrated_substrate)
    void_result = void_op.measure(calibrated_substrate)
    print(f"    Origin: {void_result['origin']}")
    print(f"    State:  {void_result['ultimate_reality']}")
    
    print("\n")
    console_log(">>> SEQUENCE COMPLETE. TARGET DISASSEMBLED.", delay=0.02)
    print("\n")

if __name__ == "__main__":
    main()
