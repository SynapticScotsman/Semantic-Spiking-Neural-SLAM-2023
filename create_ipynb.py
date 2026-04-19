import json

def convert():
    try:
        with open('experiments/test_perception.py', 'r', encoding='utf-8') as f:
            text = f.read()
            
        cells = []
        for block in text.split('# %%'):
            if not block.strip(): continue
            if block.startswith(' [markdown]'):
                lines = block.split('\n')[1:] # drop the ' [markdown]' line
                cells.append({
                    "cell_type": "markdown", 
                    "metadata": {}, 
                    "source": [l + "\n" for l in lines]
                })
            else:
                cells.append({
                    "cell_type": "code", 
                    "execution_count": None, 
                    "metadata": {}, 
                    "outputs": [], 
                    "source": [l + "\n" for l in block.strip('\n').split('\n')]
                })
                
        nb = {
            "cells": cells, 
            "metadata": {}, 
            "nbformat": 4, 
            "nbformat_minor": 4
        }
        
        with open('experiments/test_perception.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully created test_perception.ipynb!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    convert()
