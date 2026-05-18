"""Check operations that we do not consider. 
"""
from __future__ import annotations

import sys
import argparse

import onnx
from collections import Counter
from src.common.legality.checker import LegalityChecker


class OpAnalyzer:

    def __init__(self, 
                 args 
                 ) -> None:
        self.model_path = args.model_path
        return 

    
    def _get_onnx(self):
        model = onnx.load(self.model_path)
        self.model = model
        return 


    def run(self):
        return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")   
    args = parser.parse_args()  
    obj = OpAnalyzer(args)
    obj.run()
    return 


if __name__ == "__main__":
    main()